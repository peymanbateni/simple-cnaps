import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
from config_networks import ConfigureNetworks
from set_encoder import mean_pooling
import torch.nn.functional as F
import numpy as np

torch.autograd.set_detect_anomaly(True)

NUM_SAMPLES=1

class SimpleCnaps(nn.Module):
    """
    Main model class. Implements several Simple CNAPs models (with / without feature adaptation, with /without auto-regressive
    adaptation parameters generation.
    :param device: (str) Device (gpu or cpu) on which model resides.
    :param use_two_gpus: (bool) Whether to paralleize the model (model parallelism) across two GPUs.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """
    def __init__(self, device, use_two_gpus, args, mt=False):
        super(SimpleCnaps, self).__init__()
        self.args = args
        self.device = device
        self.use_two_gpus = use_two_gpus
        networks = ConfigureNetworks(pretrained_resnet_path=self.args.pretrained_resnet_path,
                                     feature_adaptation=self.args.feature_adaptation, mt=mt)
        self.set_encoder = networks.get_encoder()

        """
        Since Simple CNAPS relies on the Mahalanobis distance, it doesn't require
        the classification adaptation function and the classifier itself in CNAPS.
        The removal of the former results in a 788,485 reduction in the number of parameters
        in the model.
        """
        #self.classifier_adaptation_network = networks.get_classifier_adaptation()
        #self.classifier = networks.get_classifier()

        self.feature_extractor = networks.get_feature_extractor()
        self.feature_adaptation_network = networks.get_feature_adaptation()
        self.task_representation = None
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation

        """
        In addition to saving class representations, Simple CNAPS uses a separate
        ordered dictionary for saving the class percision matrices for use when infering on
        query examples.
        """
        self.class_precision_matrices = OrderedDict() # Dictionary mapping class label (integer) to regularized precision matrices estimated

    def forward(self, context_images, context_labels, target_images):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """

        # extract train and test features
        self.task_representation = self.set_encoder(context_images)
        context_features, target_features = self._get_features(context_images, target_images)

        """
        We build both class representations and the regularized covariance estimates.
        """
        # get the class means and covariance estimates in tensor form
        self._build_class_reps_and_covariance_estimates(context_features, context_labels)
        class_means = torch.stack(list(self.class_representations.values())).squeeze(1)
        class_precision_matrices = torch.stack(list(self.class_precision_matrices.values()))

        # grabbing the number of classes and query examples for easier use later in the function
        number_of_classes = class_means.size(0)
        number_of_targets = target_features.size(0)

        """
        Calculating the Mahalanobis distance between query examples and the class means
        including the class precision estimates in the calculations, reshaping the distances
        and multiplying by -1 to produce the sample logits
        """
        repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
        repeated_class_means = class_means.repeat(number_of_targets, 1)
        repeated_difference = (repeated_class_means - repeated_target)
        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes, repeated_difference.size(1)).permute(1, 0, 2)
        first_half = torch.matmul(repeated_difference, class_precision_matrices)
        sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1,0) * -1

        # clear all dictionaries
        self.class_representations.clear()
        self.class_precision_matrices.clear()

        return split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_images.shape[0]])

    def _get_features(self, context_images, target_images):
        """
        Helper function to extract task-dependent feature representation for each image in both context and target sets.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (tuple::torch.tensor) Feature representation for each set of images.
        """
        # Parallelize forward pass across multiple GPUs (model parallelism)
        if self.use_two_gpus:
            context_images_1 = context_images.cuda(1)
            target_images_1 = target_images.cuda(1)
            if self.args.feature_adaptation == 'film+ar':
                task_representation_1 = self.task_representation.cuda(1)
                # Get adaptation params by passing context set through the adaptation networks
                self.feature_extractor_params = self.feature_adaptation_network(context_images_1, task_representation_1)
            else:
                task_representation_1 = self.task_representation.cuda(1)
                # Get adaptation params by passing context set through the adaptation networks
                self.feature_extractor_params = self.feature_adaptation_network(task_representation_1)
            # Given adaptation parameters for task, conditional forward pass through the adapted feature extractor
            context_features_1 = self.feature_extractor(context_images_1, self.feature_extractor_params)
            context_features = context_features_1.cuda(0)
            target_features_1 = self.feature_extractor(target_images_1, self.feature_extractor_params)
            target_features = target_features_1.cuda(0)
        else:
            if self.args.feature_adaptation == 'film+ar':
                # Get adaptation params by passing context set through the adaptation networks
                self.feature_extractor_params = self.feature_adaptation_network(context_images, self.task_representation)
            else:
                # Get adaptation params by passing context set through the adaptation networks
                self.feature_extractor_params = self.feature_adaptation_network(self.task_representation)
            # Given adaptation parameters for task, conditional forward pass through the adapted feature extractor
            context_features = self.feature_extractor(context_images, self.feature_extractor_params)
            target_features = self.feature_extractor(target_images, self.feature_extractor_params)

        return context_features, target_features

    def _build_class_reps_and_covariance_estimates(self, context_features, context_labels):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """

        """
        Calculating a task level covariance estimate using the provided function.
        """
        task_covariance_estimate = self.estimate_cov(context_features)
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = mean_pooling(class_features)
            # updating the class representations dictionary with the mean pooled representation
            self.class_representations[c.item()] = class_rep
            """
            Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
            Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
            inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
            dictionary for use later in infering of the query data points.
            """
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            self.class_precision_matrices[c.item()] = torch.inverse((lambda_k_tau * self.estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
                    + torch.eye(class_features.size(1), class_features.size(1)).cuda(0))

    def estimate_cov(self, examples, rowvar=False, inplace=False):
        """
        Function based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def distribute_model(self):
        self.feature_extractor.cuda(1)
        self.feature_adaptation_network.cuda(1)
