import torch
import os
from datetime import datetime
import math


"""
print_and_log: Helper function to print to the screen and the log file.
"""


def print_and_log(log_file, message):
    print(message)
    log_file.write(message + '\n')


class ValidationAccuracies:
    def __init__(self, args):
        self.datasets = args.validation_datasets
        self.dataset_count = len(self.datasets)
        self.current_best_accuracy_dict = {}
        for dataset in self.datasets:
            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

    def is_better(self, accuracies_dict):
        is_better = False
        is_better_count = 0
        for i, dataset in enumerate(self.datasets):
            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
                is_better_count += 1

        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
            is_better = True

        return is_better

    def replace(self, accuracies_dict):
        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "Validation Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.2f}+/-{2:.2f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")

"""
get_log_files: Function that takes a path to a checkpoint directory and returns
a reference to a logfile and paths to the fully trained model and the model
with the best validation score.
"""


def get_log_files(checkpoint_dir):
    unique_checkpoint_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(unique_checkpoint_dir):
        os.makedirs(unique_checkpoint_dir)
    checkpoint_path_validation = os.path.join(unique_checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(unique_checkpoint_dir, 'fully_trained.pt')
    logfile_path = os.path.join(unique_checkpoint_dir, 'log')
    logfile = open(logfile_path, "w")

    return unique_checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final


def stack_first_dim(x):
    # method to combine the first two dimension of an array
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)


def split_first_dim(x, first_two_dims):
    # undo the stacking operation
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += x_shape[1:]
    return x.view(new_shape)


def split_first_dim_linear(x, first_two_dims):
    # undo the stacking operation
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


def class_conditional_pooling(x, y):
    class_reps = []
    for c in torch.unique(y):
        # filter out feature vectors which have class c
        class_mask = torch.eq(y, c)                            # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)         # indices of labels equal to which class
        class_mask = torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
        class_features = torch.index_select(x, 0, class_mask)
        class_reps.append(torch.mean(class_features, dim=0, keepdim=True))
    return torch.squeeze(torch.stack(class_reps))


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tf tensor - mean parameter of the distribution
    :param var: tf tensor - log variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tf tensor - samples from distribution of size numSamples x dim(mean)
    """
    # example: sample_shape = [L, 1, 1, 1, 1]
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()

def mlpip_loss(test_logits_sample, test_labels, device):
    """
    Compute the ML-PIP loss given sample logits, and test labels

    Args:
	test_logits_sample (torch.tensor): samples of logits for forward pass
        (num_samples x num_target x num_classes)
        test_labels (torch.tensor): ground truth labels
        (num_target x num_classes)
        device (torch.device): device we're running on
    Returns:
	(torch.scalar): computation of ML-PIP loss function
    """
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)


def _mlpip_loss(test_logits_sample, test_labels, device):
    """
    Compute the ML-PIP loss given sample logits, and test labels. Computes as

        log (1 / L) ∑ exp ( log p(D_t | z_l) )
            = log ∑ exp ( log p(D_t | z_l) ) - log L
            = LSE_l (log p(D_t | z_l)) - log L

    Args:
	test_logits_sample (torch.tensor): samples of logits for forward pass
        (num_samples x num_target x num_classes)
        test_labels (torch.tensor): ground truth labels
        (num_target x num_classes)
        device (torch.device): device we're running on
    Returns:
	(torch.scalar): computation of ML-PIP loss function
    """
    # Extract number of samples
    num_samples = test_logits.shape[0]

    # log p(D_t | z_l) = ∑ log p(y | x, z_l)
    log_py = -F.cross_entropy(test_logits_samples, test_labels, reduction='sum')
    # tensorize L
    l = torch.tensor([num_samples], dtype=torch.float, device=device)
    mlpips = torch.logsumexp(log_py, dim=0) - torch.log(l)
    return -mlpips.mean()


def aggregate_accuracy(test_logits_sample, test_labels):
    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())
