import torch
import torch.nn as nn
from .utils import split_first_dim_linear
from .config_networks import ConfigureNetworks

torch.autograd.set_detect_anomaly(True)

NUM_SAMPLES = 1

class TransductiveCnaps(nn.Module):
    """
    Main model class. Implements several CNAPs models (with / without feature adaptation, with /without auto-regressive
    adaptation parameters generation.
    :param device: (str) Device (gpu or cpu) on which model resides.
    :param use_two_gpus: (bool) Whether to paralleize the model (model parallelism) across two GPUs.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    :param cluster_refinement_param_mode: (str) One of 'train' or 'test', specifying which min/max set of params to use
    :param mt: (bool) If true, model is being evaluated on mini/tiered imagenet - loads ResNet checkpoints differently
    """
    def __init__(self, device, use_two_gpus, args, cluster_refinement_param_mode="test", mt=False):
        super(TransductiveCnaps, self).__init__()
        self.cluster_refinement_param_mode = cluster_refinement_param_mode
        self.args = args
        self.device = device
        self.use_two_gpus = use_two_gpus
        networks = ConfigureNetworks(pretrained_resnet_path=self.args.pretrained_resnet_path,
                                     feature_adaptation=self.args.feature_adaptation, mt=mt)
        self.set_encoder = networks.get_encoder()

        self.feature_extractor = networks.get_feature_extractor()
        self.feature_adaptation_network = networks.get_feature_adaptation()
        self.identity_coefficient = 1.0 #torch.nn.Parameter(torch.Tensor([2.0]))
        self.task_representation = None

        self.class_means = None
        self.class_precisions = None

    def set_cluster_refinement_param_mode_to_train(self):
        """
        Sets model mode to training. Max/min refinement steps set to train time specific values.
        """
        self.cluster_refinement_param_mode = "train"

    def set_cluster_refinement_param_mode_to_test(self):
        """
        Sets model mode to testing. Max/min refinement steps set to test time specific values.
        """
        self.cluster_refinement_param_mode = "test"

    def forward(self, context_images, context_labels, target_images):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """

        # one-hot encode label
        context_probs = torch.zeros(context_images.shape[0], context_labels.max() + 1, device=context_images.device)
        context_probs[range(context_labels.shape[0]), context_labels.long()] = 1.0

        # extract train and test features
        self.task_representation = self.set_encoder(context_images, context_probs, target_images)
        context_features, target_features = self._get_features(context_images, target_images)

        # First, just use supervised data to estimate class parameters, then label the unlabled data
        self.estimate_cluster_parameters(context_features, context_probs)
        target_logits = self.estimate_class_posteriors(target_features)

        # Next, run a number of cluster refinement steps to re-estimate the parameters and
        # relabel the data. If the model is in "train" mode, we do at least `min_cluster_refinement_steps_train`
        # iterations, and then break early if the labels stop changing or continue for a 'max_cluster_refinement_steps_train'
        # steps. If the model is in "test" mode, test mode step paramters are used.
        if self.cluster_refinement_param_mode == "train":
            if self.args.max_cluster_refinement_steps_train > 0:
                combined_features = torch.cat([context_features, target_features], dim=0)
                for step in range(self.args.max_cluster_refinement_steps_train):
                    # Update cluster parameters based on soft assignments
                    combined_probs = torch.cat([context_probs, nn.functional.softmax(target_logits, dim=1)], dim=0)
                    self.estimate_cluster_parameters(combined_features, combined_probs)
                    old_logits = target_logits
                    target_logits = self.estimate_class_posteriors(target_features)

                    if step >= self.args.min_cluster_refinement_steps_train-1:
                        delta_l = (old_logits.argmax(dim=1) != target_logits.argmax(dim=1)).sum().detach().cpu().item()
                        if delta_l == 0:
                            break
        elif self.cluster_refinement_param_mode == "test":
            if self.args.max_cluster_refinement_steps_test > 0:
                combined_features = torch.cat([context_features, target_features], dim=0)
                for step in range(self.args.max_cluster_refinement_steps_test):
                    # Update cluster parameters based on soft assignments
                    combined_probs = torch.cat([context_probs, nn.functional.softmax(target_logits, dim=1)], dim=0)
                    self.estimate_cluster_parameters(combined_features, combined_probs)
                    old_logits = target_logits
                    target_logits = self.estimate_class_posteriors(target_features)

                    if step >= self.args.min_cluster_refinement_steps_test-1:
                        delta_l = (old_logits.argmax(dim=1) != target_logits.argmax(dim=1)).sum().detach().cpu().item()
                        if delta_l == 0:
                            break
        else:
            raise Exception("Incorrect model mode has been set. Please choose one of 'train' or 'test'.")

        return split_first_dim_linear(target_logits, [NUM_SAMPLES, target_images.shape[0]])

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

    def estimate_cluster_parameters(self, features, labels_soft):
        """
        Inputs:
        `features`:    [instance, features] tensor
        `labels_soft`: [instance, class] tensor of class probabilities

        Estimate class covariances given (soft) class assignments, using the shrinkage
        estimator defined in the SimpleCNAPS paper.
        """
        # Global covariance
        centered = (features - features.mean(dim=0, keepdim=True))
        sigma0 = torch.einsum('if,ig->fg', centered, centered)/(features.shape[0] - 1)

        # Per-class means
        N = labels_soft.sum(dim=0) # Per-class "effective samples"
        mu = torch.einsum('if,ic->cf', features, labels_soft) / N[:, None]
        centered = features[:, None, :] - mu[None, :, :] # (instance, class, feature)
        S = torch.einsum('ic,icf,icg->cfg', labels_soft, centered, centered) / N[:, None, None]

        # Shrinkage + diagonal loading
        lamb = (N / (N + 1.0))[:, None, None] # Shrinkage weight
        I = torch.eye(features.shape[-1], device=features.device)[None, :, :]
        S = (self.identity_coefficient * I) + lamb*S + (1.0 - lamb)*sigma0

        self.class_means = mu
        self.class_precisions = torch.inverse(S)

    def estimate_class_posteriors(self, features):
        """
        Inputs:
        `features`: (instance, feature) tensor
        Returns:
        (instance, class) tensor of logits
        """
        # centered is (instance, class, feature)
        centered = features[:, None, :] - self.class_means[None, :, :]
        # (instance, class) containing posterior logits
        posterior_logits = -1*torch.einsum('cfg,icf,icg->ic', self.class_precisions, centered, centered)

        return posterior_logits

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def distribute_model(self):
        self.feature_extractor.cuda(1)
        self.feature_adaptation_network.cuda(1)
