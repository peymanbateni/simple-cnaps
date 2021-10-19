from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import argparse
import os
import pickle

from utils import print_and_log, get_log_files, stack_first_dim, mlpip_loss, aggregate_accuracy
from simple_cnaps.simple_cnaps import SimpleCnaps
from transductive_cnaps.transductive_cnaps import TransductiveCnaps
from image_transformations import mnist_transforms, standard_transforms, cifar_transforms, cifar_transforms_no_resize
import torchvision

def load_mnist_queue(mode, batch_size):
    data = torchvision.datasets.MNIST(
        root='../data/mnist',
        train=(mode == 'train'),
        download=True,
        transform=mnist_transforms,
    )
    return torch.utils.data.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

def load_cifar10_queue(mode, batch_size, transform=cifar_transforms):
    data = torchvision.datasets.CIFAR10(
        root='../data/cifar10',
        train=(mode == 'train'),
        download=True,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

def load_cifar100_queue(mode, batch_size, transform=cifar_transforms):
    data = torchvision.datasets.CIFAR100(
        root='../data/cifar100',
        train=(mode == 'train'),
        download=True,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

def main():
    ol = ContinualLearner()
    ol.run()

def format_continual_task(train_queue, test_queue, shot, test_shot, dataset=None):
    # return task_dict with keys train_images', 'train_labels', 'test_images', 'test_labels' as before,
    # but these are now lists of length "continual time steps" with the training and test data at each time step
    train_images, train_labels = next(iter(train_queue))
    test_images, test_labels = next(iter(test_queue))
    task_dict = {
        'train_images': [],
        'train_labels': [],
        'test_images': [],
        'test_labels': [],
    }

    if dataset == 'CIFAR100':
        task_list = [[_ for _ in range(10)],
                     [_ for _ in range(10, 20)],
                     [_ for _ in range(20, 30)],
                     [_ for _ in range(30, 40)],
                     [_ for _ in range(40, 50)],
                     [_ for _ in range(50, 60)],
                     [_ for _ in range(60, 70)],
                     [_ for _ in range(70, 80)],
                     [_ for _ in range(80, 90)],
                     [_ for _ in range(90, 100)]]

    else:
        task_list = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    for task in task_list:
        task_train_images, task_train_labels = [], []
        task_test_images, task_test_labels = [], []
        for cls in task:
            cls_train_images = torch.index_select(train_images, 0, extract_class_indices(train_labels, cls))
            cls_train_labels = torch.index_select(train_labels, 0, extract_class_indices(train_labels, cls))
            cls_test_images = torch.index_select(test_images, 0, extract_class_indices(test_labels, cls))
            cls_test_labels = torch.index_select(test_labels, 0, extract_class_indices(test_labels, cls))
            task_train_images.append(cls_train_images[:shot])
            task_train_labels.append(cls_train_labels[:shot])
            task_test_images.append(cls_test_images[:test_shot])
            task_test_labels.append(cls_test_labels[:test_shot])
        task_dict['train_images'].append(torch.cat(task_train_images, dim=0))
        task_dict['train_labels'].append(torch.cat(task_train_labels, dim=0))
        task_dict['test_images'].append(torch.cat(task_test_images, dim=0))
        task_dict['test_labels'].append(torch.cat(task_test_labels, dim=0))
    return task_dict


class ContinualLearner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.accuracy_fn = aggregate_accuracy

    def init_model(self):
        if self.args.model == 'simple_cnaps':
            model = SimpleCnaps(
                device=self.device,
                use_two_gpus=True,
                args=self.args
            ).to(self.device)
            model.distribute_model()
            return model
        elif self.args.model == 'transductive_cnaps':
            model = TransductiveCnaps(
                device=self.device,
                use_two_gpus=True,
                args=self.args,
            ).to(self.device)
            model.distribute_model()
            return model
        else:
            raise Exception("Model should be one of 'simple_cnaps' or 'transductive_cnaps', but unkown model type was given.")

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        # Dataset options
        parser.add_argument("--dataset", "-d", choices=["MNIST", "CIFAR10", "CIFAR100"],
                            default="MNIST", help="Dataset to use.")
        parser.add_argument("--data_path", default="../data",
                            help="Path to image directory.")
        parser.add_argument("--batch_size_mnist", type=int, default=256, help="Batch size used on MNIST when loading the dataset and creating continual tasks.")
        parser.add_argument("--batch_size_cifar10", type=int, default=256, help="Batch size used on CIFAR10 when loading the dataset and creating continual tasks.")
        parser.add_argument("--batch_size_cifar100", type=int, default=1024, help="Batch size used on CIFAR100 when loading the dataset and creating continual tasks.")

        # Continual learning parameters
        parser.add_argument("--continual_time_steps", type=int, default=5,
                            help="Number of steps in continual learning experiment.")
        parser.add_argument("--test_epochs", type=int, default=10,
                            help="Number of testing epochs.")
        parser.add_argument("--shot", type=int, default=5,
                            help="Number of meta-training examples for training.")
        parser.add_argument("--test_shot", type=int, default=100,
                            help="Number of meta-training examples for test.")
        parser.add_argument("--results_dir", default='./continual_tests',
                            help="Number of meta-training examples for test.")
        parser.add_argument('--head_type',  default='multi',
                            help="Single or Multi type head.", choices=["single", "multi"])
        parser.add_argument("--CIFAR_resize", default='on',
                            help="Whether or not to use memory on the body.", choices=["on", "off"])
        parser.add_argument("--test_way", type=int, default=None,
                            help="Way to be used at evaluation time. If not specified 'way' will be used.")
        parser.add_argument("--way", type=int, default=5, help="Number of classes.")
        parser.add_argument("--body_memory", default='on',
                            help="Whether or not to use memory on the body.", choices=["on", "off"])

        # Model options
        parser.add_argument('--model', choices=["simple_cnaps", "transductive_cnaps"], default="simple_cnaps",
                            help="Choose model for active learning (one of 'simple_cnaps' or 'transductive_cnaps')")
        parser.add_argument("--num_labeled", type=int, default=5,
                            help="Number of labeled instances")
        parser.add_argument("--pretrained_resnet_path", default="../model-checkpoints/pretrained_resnets/pretrained_resnet_meta_dataset.pt.tar",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--test_model_path", "-m", default="../model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt", help="Path to model to load and test.")
        parser.add_argument("--checkpoint_dir", "-c", default='./checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film", "film+ar"], default="film",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument("--min_cluster_refinement_steps_test", type=int, default=2, help="Minimum number of soft-kmeans clustering steps.")
        parser.add_argument("--max_cluster_refinement_steps_test", type=int, default=4, help="Maximum number of soft-kmeans clustering steps.")
        parser.add_argument("--continual_learning_strategy", choices=["moving-encoding", "first-encoding", "averaging-encoding"], default="averaging-encoding",
                            help="Continual learning strategy to use for updating the task encoding.")

        args = parser.parse_args()

        # adjust test_shot and test_way if necessary
        if args.test_shot is None:
            args.test_shot = args.shot
        if args.test_way is None:
            args.test_way = args.way

        return args

    def run(self):
        self.test(self.args.test_model_path, self.args.results_dir)
        self.logfile.close()

    def map_labels(self, labels, device):
        map_dict = {}
        labels = labels.cpu().numpy().copy()
        for new_label, old_label in enumerate(np.sort(np.unique(labels))):
            map_dict[old_label] = new_label
        for i in range(len(labels)):
            labels[i] = map_dict[labels[i]]
        return torch.from_numpy(labels).type(torch.LongTensor).to(device)

    def test(self, path, save_dir):
        accuracies = []

        if self.args.dataset == 'MNIST':
            train_queue = load_mnist_queue('train', self.args.batch_size_mnist)
            test_queue = load_mnist_queue('test', self.args.batch_size_mnist)
        elif self.args.dataset == 'CIFAR10':
            if self.args.CIFAR_resize == 'off':
                train_queue = load_cifar10_queue('train', self.args.batch_size_cifar10, transform=cifar_transforms_no_resize)
                test_queue = load_cifar10_queue('test', self.args.batch_size_cifar10, transform=cifar_transforms_no_resize)
            else:
                train_queue = load_cifar10_queue('train', self.args.batch_size_cifar10)
                test_queue = load_cifar10_queue('test', self.args.batch_size_cifar10)
        elif self.args.dataset == 'CIFAR100':
            if self.args.CIFAR_resize == 'off':
                train_queue = load_cifar100_queue('train', self.args.batch_size_cifar100, transform=cifar_transforms_no_resize)
                test_queue = load_cifar100_queue('test', self.args.batch_size_cifar100, transform=cifar_transforms_no_resize)
            else:
                train_queue = load_cifar100_queue('train', self.args.batch_size_cifar100)
                test_queue = load_cifar100_queue('test', self.args.batch_size_cifar100)

        for epoch in range(self.args.test_epochs):
            # load the model
            self.model = self.init_model()
            self.model.load_state_dict(torch.load(path))
            self.model.eval()

            if (epoch+1) % 10 == 0:
                print('Test epoch {} on {} using {} with {} strategy.'.format((epoch + 1), self.args.dataset, 
                    self.args.model, self.args.continual_learning_strategy))
            acc_for_epoch = np.empty((self.args.continual_time_steps, self.args.continual_time_steps))
            acc_for_epoch[:] = np.nan
            task_dict = format_continual_task(
                train_queue=train_queue,
                test_queue=test_queue,
                shot=self.args.shot,
                test_shot=self.args.test_shot,
                dataset=self.args.dataset
            )

            all_train_images, all_train_labels = task_dict['train_images'], task_dict['train_labels']
            all_test_images, all_test_labels = task_dict['test_images'], task_dict['test_labels']

            for t in range(self.args.continual_time_steps):
                if self.args.body_memory == 'off':
                    self.model.clear_task_rep_count()
                
                # train the model with the new training data and test on the new task
                train_images_time_step = all_train_images[t].to(self.device)
                train_labels_time_step = all_train_labels[t].to(self.device)
                test_images_time_step = all_test_images[t].to(self.device)
                test_labels_time_step = all_test_labels[t].type(torch.LongTensor).to(self.device)

                if self.args.head_type == 'multi':
                    label_set = torch.unique(test_labels_time_step, sorted=True)
                    label_set = [l.item() for l in label_set]
                    test_labels_mapped = self.map_labels(test_labels_time_step, self.device)
                elif self.args.head_type == 'single':
                    label_set = torch.cat([all_test_labels[d] for d in range(t+1)]).type(torch.LongTensor).to(self.device)
                    label_set = torch.unique(label_set, sorted=True)
                    label_set = [l.item() for l in label_set]
                    test_labels_mapped = test_labels_time_step

                
                test_logits_time_step = self.model(train_images_time_step, train_labels_time_step, test_images_time_step, label_set=label_set)

                accuracy = self.accuracy_fn(test_logits_time_step, test_labels_mapped)
                acc_for_epoch[t, t] = accuracy.item()

                all_train_images[t] = all_train_images[t].cpu()
                all_train_labels[t] = all_train_labels[t].cpu()
                all_test_images[t] = all_test_images[t].cpu()
                all_test_labels[t] = all_test_labels[t].cpu()

                train_images_time_step = train_images_time_step.cpu()
                train_labels_time_step = train_labels_time_step.cpu()
                test_images_time_step = test_images_time_step.cpu()
                test_labels_time_step = test_labels_time_step.cpu()

                del test_logits_time_step

                for i in range(t):
                    # test on all of the old tasks from previous time steps
                    test_images_time_step = all_test_images[i].to(self.device)
                    test_labels_time_step = all_test_labels[i].type(torch.LongTensor).to(self.device)
                    if self.args.head_type == 'multi':
                        label_set = torch.unique(test_labels_time_step, sorted=True)
                        label_set = [l.item() for l in label_set]
                    elif self.args.head_type == 'single':
                        pass

                    test_logits_time_step = self.model(None, None, test_images_time_step, label_set=label_set)
                    if self.args.head_type == 'multi':
                        test_labels_mapped = self.map_labels(test_labels_time_step, self.device)
                    else:
                        test_labels_mapped = test_labels_time_step

                    accuracy = self.accuracy_fn(test_logits_time_step, test_labels_mapped)
                    acc_for_epoch[t, i] = accuracy.item()

                    all_test_images[i] = all_test_images[i].cpu()
                    all_test_labels[i] = all_test_labels[i].cpu()

                    test_images_time_step = test_images_time_step.cpu()
                    test_labels_time_step = test_labels_time_step.cpu()

                    del test_logits_time_step

            accuracies.append(acc_for_epoch)

            # clear representations
            if self.args.model == 'simple_cnaps':
                self.model.class_representations.clear()  
                self.model.class_precision_matrices.clear() 
            elif self.args.model == 'transductive_cnaps':
                self.model.class_means.clear()
                self.model.class_precisions.clear()
            self.model.task_counter = 0.0

            #all_train_images, all_train_labels = all_train_images.cpu(), all_train_labels.cpu()
            #all_test_images, all_test_labels = all_test_images.cpu(), all_test_labels.cpu()

            del all_train_images, all_train_labels, all_test_images, all_test_labels

        accuracies = np.array(accuracies)
        save_path = os.path.join(save_dir,
                                 'continual_results_{}_shot_{}_testshot_{}_testepochs_{}_headtype_{}_for_{}_with_{}.pickle'.format(self.args.dataset,
                                                              self.args.shot,
                                                              self.args.test_shot,
                                                              self.args.test_epochs,
                                                              self.args.head_type,
                                                              self.args.model,
                                                              self.args.continual_learning_strategy))
        # getting the model off the GPU memory
        self.model = self.model.cpu()
        del self.model

        with open(save_path, 'wb') as handle:
            pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
