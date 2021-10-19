from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import numpy as np
import argparse
import os
import pickle
import random

from utils import print_and_log, get_log_files, stack_first_dim, split_first_dim_linear, ValidationAccuracies, mlpip_loss, aggregate_accuracy
from image_transformations import omniglot_transforms, cifar_transforms, standard_transforms
from simple_cnaps.simple_cnaps import SimpleCnaps
from transductive_cnaps.transductive_cnaps import TransductiveCnaps

def load_cifar10_queue(mode, batch_size):
    data = torchvision.datasets.CIFAR10(
        root='./cifar',
        train=(mode == 'train'),
        download=True,
        transform=cifar_transforms,
    )
    return torch.utils.data.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

def get_al_data_omnigot(args, language, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(0)
    data_path = os.path.join(args.data_path, 'omniglot/test/', language)
    data = torchvision.datasets.ImageFolder(data_path, transform=omniglot_transforms)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=256,
                                              shuffle=True,
                                              num_workers=4)
    images, labels = [], []
    for step, (batch_images, batch_labels) in enumerate(data_loader):
        images.append(batch_images)
        labels.append(batch_labels)
    images, labels = torch.cat(images), torch.cat(labels)
    # separate into D0 and pool set
    D0_images, D0_labels, pool_images, pool_labels, test_images, test_labels = \
        generate_initial_sets(images, labels, args, test_size=5)
    # shuffle training and pool sets
    perm = np.random.permutation(len(D0_images))
    D0_images, D0_labels = D0_images[perm], D0_labels[perm]
    perm = np.random.permutation(len(pool_images))
    pool_images, pool_labels = pool_images[perm], pool_labels[perm]
    return D0_images, D0_labels, pool_images, pool_labels, test_images, test_labels

def get_al_data_cifar10(args):
    data_loader = load_cifar10_queue('train', 256)
    images, labels = [], []
    for step, (batch_images, batch_labels) in enumerate(data_loader):
        images.append(batch_images)
        labels.append(batch_labels)
    images, labels = torch.cat(images), torch.cat(labels)
    # separate into D0 and pool set
    D0_images, D0_labels, pool_images, pool_labels = generate_initial_sets_cifar10(images, labels, args)
    # shuffle training and pool sets
    perm = np.random.permutation(len(D0_images))
    D0_images, D0_labels = D0_images[perm], D0_labels[perm]
    perm = np.random.permutation(len(pool_images))
    pool_images, pool_labels = pool_images[perm], pool_labels[perm]
    test_data_loader = load_cifar10_queue('test', 256)
    return D0_images, D0_labels, pool_images, pool_labels, test_data_loader

def generate_initial_sets_cifar10(images, labels, args):
    train_images, train_labels = [], []
    pool_images, pool_labels = [], []
    for cls in torch.unique(labels):
        cls_images = torch.index_select(images, 0, extract_class_indices(labels, cls))
        cls_labels = torch.index_select(labels, 0, extract_class_indices(labels, cls))
        cls_n = len(cls_images)
        train_images.append(cls_images[:args.num_labeled])
        pool_images.append(cls_images[args.num_labeled:])
        train_labels.append(cls_labels[:args.num_labeled])
        pool_labels.append(cls_labels[args.num_labeled:])
    return torch.cat(train_images), torch.cat(train_labels), torch.cat(pool_images), torch.cat(pool_labels)

def generate_initial_sets(images, labels, args, test_size):
    train_images, train_labels = [], []
    pool_images, pool_labels = [], []
    test_images, test_labels = [], []
    for cls in torch.unique(labels):
        cls_images = torch.index_select(images, 0, extract_class_indices(labels, cls))
        cls_labels = torch.index_select(labels, 0, extract_class_indices(labels, cls))
        cls_n = len(cls_images)
        train_images.append(cls_images[:args.num_labeled])
        pool_images.append(cls_images[args.num_labeled:cls_n - test_size])
        test_images.append(cls_images[-test_size:])
        train_labels.append(cls_labels[:args.num_labeled])
        pool_labels.append(cls_labels[args.num_labeled:cls_n - test_size])
        test_labels.append(cls_labels[-test_size:])
    return torch.cat(train_images), torch.cat(train_labels), \
           torch.cat(pool_images), torch.cat(pool_labels), \
           torch.cat(test_images), torch.cat(test_labels)

def balance_labeled_set(images, labels, args):
    balanced_images, balanced_labels = [], []
    for cls in torch.unique(labels):
        cls_images = torch.index_select(images, 0, extract_class_indices(labels, cls))
        cls_labels = torch.index_select(labels, 0, extract_class_indices(labels, cls))
        balanced_images.append(cls_images[:args.num_labeled])
        balanced_labels.append(cls_labels[:args.num_labeled])
    return torch.cat(balanced_images), torch.cat(balanced_labels)

def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

def main():
    ol = ActiveLearner()
    ol.run()

class ActiveLearner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.LANGUAGES = os.listdir(os.path.join(self.args.data_path, 'omniglot/test/'))
        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        if self.args.model in ['simple_cnaps', 'transductive_cnaps']:
            self.model.distribute_model()
        self.accuracy_fn = aggregate_accuracy

    def init_model(self):
        if self.args.model == 'simple_cnaps':
            model = SimpleCnaps(
                device=self.device,
                use_two_gpus=True,
                args=self.args,
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

        # Active learning parameters
        parser.add_argument("--validation_size", type=int, default=10000, help="Validation size for scoring pool set")
        parser.add_argument("--subset_size", type=int, default=None, help="Validation size for scoring pool set")
        parser.add_argument('--active_learning_method', help='What active learning acquisition function to use',
                            choices=["random", "predictive_entropy", "var_ratios"], default="random")

        # Dataset options
        parser.add_argument("--dataset", default=None, choices=['cifar10', 'omniglot'])
        parser.add_argument("--data_path", default="../dataset", help="Path to dataset directory")
        parser.add_argument("--language", default=None, help="Omniglot language to actively learn")
        parser.add_argument("--active_learning_iterations", type=int, default=30, help="Number of model updates")
        parser.add_argument("--query_set_size", type=int, default=25, help="Number of model updates")

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

        args = parser.parse_args()
        return args

    def run(self):
        if self.args.dataset == 'cifar10':
            dataset = self.args.dataset
            results_dict = {dataset: []}
            for i in range(10):
                print('method={}, dataset={}, iteration={}'.format(self.args.active_learning_method, 'cifar10', i))
                train_images, train_labels, pool_images, pool_labels, test_data_loader = \
                    get_al_data_cifar10(self.args)
                results_dict[dataset].append(self.test_cifar10(
                    path=self.args.test_model_path,
                    train_images=train_images,
                    train_labels=train_labels,
                    pool_set_images=pool_images,
                    pool_set_labels=pool_labels,
                    test_data_loader=test_data_loader
                ))
            save_dir = os.path.join(self.checkpoint_dir,
                                    self.args.active_learning_method + '_' + dataset + '.pkl')
            with open(save_dir, 'wb') as f:
                pickle.dump(results_dict, f)

        else:  # OMNIGLOT
            if self.args.language != 'all':
                language = self.args.language
                method = self.args.active_learning_method
                model = self.args.model
                results_dict = {language: []}
                for i in range(30):
                    print('method={}, dataset={}, language={}, iteration={}'.format(method, 'omniglot', language, i))
                    train_images, train_labels, pool_images, pool_labels, test_images, test_labels = \
                        get_al_data_omnigot(self.args, language, seed=i)
                    results_dict[language].append(self.test(
                        path=self.args.test_model_path,
                        train_images=train_images,
                        train_labels=train_labels,
                        pool_set_images=pool_images,
                        pool_set_labels=pool_labels,
                        test_images=test_images,
                        test_labels=test_labels
                    ))
                save_dir = os.path.join(self.checkpoint_dir, self.args.active_learning_method + '_' + language + '_' + model +'.pkl')
                with open(save_dir, 'wb') as f:
                    pickle.dump(results_dict, f)
            else:  # run the simple methods on all datasets
                model = self.args.model
                for method in ['random', 'predictive_entropy', 'var_ratios']:
                    self.args.active_learning_method = method
                    for language in self.LANGUAGES:
                        results_dict = {language: []}
                        iters = 30
                        if method == 'posterior_entropy':
                            iters = 3
                        for i in range(iters):
                            print('method={}, dataset={}, language={}, iteration={}'.format(method, 'omniglot', language, i))
                            train_images, train_labels, pool_images, pool_labels, test_images, test_labels = \
                                get_al_data_omnigot(self.args, language, seed=i)
                            results_dict[language].append(self.test(
                                path=self.args.test_model_path,
                                train_images=train_images,
                                train_labels=train_labels,
                                pool_set_images=pool_images,
                                pool_set_labels=pool_labels,
                                test_images=test_images,
                                test_labels=test_labels
                            ))
                        save_dir = os.path.join(self.checkpoint_dir,
                                                self.args.active_learning_method + '_' + language + '_' + model +'.pkl')
                        with open(save_dir, 'wb') as f:
                            pickle.dump(results_dict, f)

        self.logfile.close()

    def test_cifar10(self, path, train_images, train_labels, pool_set_images, pool_set_labels, test_data_loader):
        test_accuracies = []
        softmax = torch.nn.Softmax()

        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        torch.set_grad_enabled(False)

        for iteration in range(self.args.active_learning_iterations):
            # compute accuracy on test set with current labeled set
            import time
            iter_start_time = time.time()
            test_accuracies.append(self.compute_accuracy_cifar10(
                labeled_set=train_images,
                train_labels=train_labels,
                test_data_loader=test_data_loader
            ))

            if self.args.active_learning_method == 'predictive_entropy':
                train_images_gpu = train_images.to(self.device)
                train_labels_gpu = train_labels.to(self.device)
                pool_image_batches = torch.split(pool_set_images, 256)
                with torch.no_grad():
                    encodings = []
                    for batch in pool_image_batches:
                        batch = batch.to(self.device)
                        pool_set_logits = self.model(train_images_gpu, train_labels_gpu, batch)
                        encodings.append(pool_set_logits.squeeze())
                    logits_all = torch.cat(encodings)
                    scores = -torch.distributions.Categorical(logits=logits_all).entropy()

            elif self.args.active_learning_method == 'var_ratios':
                train_images_gpu = train_images.to(self.device)
                train_labels_gpu = train_labels.to(self.device)
                pool_image_batches = torch.split(pool_set_images, 256)
                with torch.no_grad():
                    encodings = []
                    for batch in pool_image_batches:
                        batch = batch.to(self.device)
                        pool_set_logits = self.model(train_images_gpu, train_labels_gpu, batch)
                        encodings.append(pool_set_logits.squeeze())
                    logits_all = torch.cat(encodings)
                    py = softmax(logits_all)
                    scores = -(1. - torch.max(py, dim=1)[0])

            elif self.args.active_learning_method == 'random':
                scores = torch.rand(len(pool_set_images))

            # Move x* from pool queue to labeled set (with label)
            for _ in range(self.args.query_set_size):
                j_star = torch.argmin(scores, 0)
                train_images, train_labels, pool_set_images, pool_set_labels = self.add_index_to_set_cifar10(
                    images=train_images,
                    labels=train_labels,
                    pool_images=pool_set_images,
                    pool_labels=pool_set_labels,
                    index=j_star
                )
                # delete j_star from scores
                scores = torch.cat([scores[:j_star], scores[j_star + 1:]])
            iter_time = time.time() - iter_start_time
            print('Finished active learning iteration: %s, accuracy: %s, time: % s'
                  % (iteration, test_accuracies[-1], iter_time))

        return np.array(test_accuracies)

    def compute_accuracy_cifar10(self, labeled_set, train_labels, test_data_loader):
        train_images_gpu = labeled_set.to(self.device)
        train_labels_gpu = train_labels.to(self.device)
        encodings = []
        test_labels_list = []
        for test_images, test_labels in test_data_loader:
            test_images = test_images.to(self.device)
            test_labels = test_labels.to(self.device)
            test_labels_list.append(test_labels)
            test_logits = self.model(train_images_gpu, train_labels_gpu, test_images)
            encodings.append(test_logits.squeeze())
        logits_all = torch.cat(encodings)
        test_labels_all = torch.cat(test_labels_list)
        accuracy = self.accuracy_fn(logits_all.unsqueeze(0), test_labels_all)
        return accuracy.item()

    def test(self, path, train_images, train_labels, pool_set_images, pool_set_labels, test_images, test_labels):
        test_accuracies = []
        softmax = torch.nn.Softmax()

        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        torch.set_grad_enabled(False)

        train_images = train_images.to(self.device)
        train_labels = train_labels.to(self.device)
        pool_set_images = pool_set_images.to(self.device)
        pool_set_labels = pool_set_labels.to(self.device)
        test_images = test_images.to(self.device)
        test_labels = test_labels.to(self.device)

        for iteration in range(self.args.active_learning_iterations):
            # compute accuracy on test set with current labeled set
            import time
            iter_start_time = time.time()
            test_accuracies.append(self.compute_accuracy(
                labeled_set=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels
            ))

            # compute p(y | x, D) for every x in pool set
            if self.args.active_learning_method != 'random':
                pool_set_batches = torch.split(pool_set_images, 256)
                pool_set_logits = [self.model(train_images, train_labels, batch) for batch in
                     pool_set_batches]
                pool_set_logits = torch.cat(pool_set_logits, dim=1)

            if self.args.active_learning_method == 'predictive_entropy':
                scores = -torch.distributions.Categorical(logits=pool_set_logits.squeeze()).entropy()

            elif self.args.active_learning_method == 'var_ratios':
                # pool_set_logits, _ = self.model(train_images, train_labels, pool_set_images, print_gradients=False)
                py = softmax(pool_set_logits.squeeze())
                scores = -(1. - torch.max(py, dim=1)[0])

            elif self.args.active_learning_method == 'random':
                scores = torch.rand(len(pool_set_images))

            # Move x* from pool queue to labeled set (with label)
            j_star = torch.argmin(scores, 0)
            train_images, train_labels, pool_set_images, pool_set_labels = self.add_index_to_set(
                images=train_images,
                labels=train_labels,
                pool_images=pool_set_images,
                pool_labels=pool_set_labels,
                index=j_star
            )
            iter_time = time.time() - iter_start_time
            print('Finished active learning iteration: %s, accuracy: %s, time: % s'
                  % (iteration, test_accuracies[-1], iter_time))

        return np.array(test_accuracies)

    def compute_accuracy(self, labeled_set, train_labels, test_images, test_labels):
        test_logits = self.model(labeled_set, train_labels, test_images)
        accuracy = self.accuracy_fn(test_logits, test_labels)
        return accuracy.item()

    def compute_entropy_score(self, py, image, train_images, train_labels, val_images):
        s = 0
        train_images_prime = torch.cat([train_images, image[None, :]], dim=0)
        sorted_labels = torch.argsort(py, descending=True)
        for label in sorted_labels[:3]:
            y_prime = torch.LongTensor([label, ]).to(self.device)
            train_labels_prime = torch.cat([train_labels, y_prime], dim=0)
            val_logits = self.model(train_images_prime, train_labels_prime, val_images)
            predictive_entropies = torch.distributions.Categorical(logits=val_logits).entropy()
            average_entropy = predictive_entropies.mean()
            s += py[label] * average_entropy
        return s

    def add_index_to_set(self, images, labels, pool_images, pool_labels, index):
        images = torch.cat([images, pool_images[index][None, :].cuda()], dim=0)
        labels = torch.cat([labels, pool_labels[index].unsqueeze(0).cuda()], dim=0)
        pool_images = torch.cat([pool_images[0:index], pool_images[index + 1:]])
        pool_labels = torch.cat([pool_labels[0:index], pool_labels[index + 1:]])
        return images, labels, pool_images, pool_labels

    def add_index_to_set_cifar10(self, images, labels, pool_images, pool_labels, index):
        images = torch.cat([images, pool_images[index][None, :]], dim=0)
        labels = torch.cat([labels, pool_labels[index].unsqueeze(0)], dim=0)
        pool_images = torch.cat([pool_images[0:index], pool_images[index + 1:]])
        pool_labels = torch.cat([pool_labels[0:index], pool_labels[index + 1:]])
        return images, labels, pool_images, pool_labels

if __name__ == "__main__":
    main()
