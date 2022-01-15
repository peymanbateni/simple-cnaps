# Enhancing Few-Shot Image Classification with Unlabelled Examples

This directory contains the code for the paper, "Enhancing Few-Shot Image Classification with Unlabelled Examples", which has been published at IEEE WACV 2022. For a pdf copy of the paper, please visit our ArXiv copy at https://arxiv.org/pdf/2006.12245.pdf.

Global Meta-Dataset Rank (Transductive CNAPS): https://github.com/google-research/meta-dataset#training-on-all-datasets

Global Mini-ImageNet Rank (Transductive CNAPS):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-few-shot-visual-classification-with/few-shot-image-classification-on-mini-2)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-2?p=improving-few-shot-visual-classification-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-few-shot-visual-classification-with/few-shot-image-classification-on-mini-3)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-3?p=improving-few-shot-visual-classification-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-few-shot-visual-classification-with/few-shot-image-classification-on-mini-12)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-12?p=improving-few-shot-visual-classification-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-few-shot-visual-classification-with/few-shot-image-classification-on-mini-13)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-13?p=improving-few-shot-visual-classification-with)

Global Tiered-ImageNet Rank (Transductive CNAPS):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-few-shot-visual-classification-with/few-shot-image-classification-on-tiered)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered?p=improving-few-shot-visual-classification-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-few-shot-visual-classification-with/few-shot-image-classification-on-tiered-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered-1?p=improving-few-shot-visual-classification-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-few-shot-visual-classification-with/few-shot-image-classification-on-mini-12)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-12?p=improving-few-shot-visual-classification-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-few-shot-visual-classification-with/few-shot-image-classification-on-mini-13)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-13?p=improving-few-shot-visual-classification-with)

## Dependencies
You can use the ```requirements.txt``` file included to install dependencies for both Simple CNAPS, Transductive CNAPS, relevant active/continual learning experiments and the accompanying source codes for Meta-Dataset, mini-ImageNet and tiered-ImageNet. To install all dependencies, run ```pip install -r requirements.txt```. In general, this code base requires Python 3.5 or greater, PyTorch 1.0 or greater and TensorFlow 2.0 or greater.

## GPU Requirements
The GPU requirements for Simple CNAPS are:
* 1 GPU with 16GB or more memory for training Simple CNAPS - you can alternatively perform distributed training of Simple CNAPS across 2 GPUs with 8GB or more in dedicated memory (we primarily used this setting in our experiments)
* 2 GPUs with 16GB or more memory for training Simple AR-CNAPS

We recommend the same settings for testing, although as gradient propagation is no more required, you may be able to test on GPUs with much less memory.

## Meta-Dataset Installation
Our installation process is the same as CNAPS:
1. Clone or download this repository.
2. Configure Meta-Dataset:
    * Note that as of writing, we have updated our code to work with the latest version of the Meta-Dataset repository. We have included a copy of this repository under the ```meta-dataset``` folder for ease of use and consistency, should there be major subsequent updates to the code-base. 
    * Please follow the "User instructions" in the Meta-Dataset repository (visit https://github.com/google-research/meta-dataset or the ```meta-dataset``` folder)
    for "Installation" and "Downloading and converting datasets". This step can take up-to 48 hours.
3. Install additional test datasets (MNIST, CIFAR10, CIFAR100):
    * Change to the $DATASRC directory: ```cd $DATASRC```
    * Download the MNIST test images: ```wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz```
    * Download the MNIST test labels: ```wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz```
    * Download the CIFAR10 dataset: ```wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz```
    * Extract the CIFAR10 dataset: ```tar -zxvf cifar-10-python.tar.gz```
    * Download the CIFAR100 dataset: ```wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz```
    * Extract the CIFAR10 dataset: ```tar -zxvf cifar-100-python.tar.gz```
    * Change to the ```simple-cnaps/transductive-cnaps-src/``` directory in the repository.
    * Run: ```python prepare_extra_datasets.py```

## Meta-Dataset Usage
All test scripts used to produce results reported within the paper have been provided in the [test-scripts](https://github.com/plai-group/simple-cnaps/tree/master/transductive-cnaps-src/test-scripts/) directory. To train and test Transductive CNAPs on Meta-Dataset:

1. First run the following three commands:
    
    ```ulimit -n 50000```
    
    ```export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>```
    
    ```export RECORDS=<root directory of where you have produced meta-dataset records>```
    
    Note that you may need to run the above commands every time you open a new command shell.
    
2. We have provided our Transductive CNAPS model checkpoint, correspondingly named "best_transductive_cnaps.pt", under the [model-checkpoints](https://github.com/plai-group/simple-cnaps/tree/master/model-checkpoints/meta-dataset-checkpoints/) folder. This checkpoint contains the trained parameters for the model that produced the SoTA results for Transductive CNAPS (as referenced in the paper). To re-run evaluation, you can use the following command to test the provided Transductive CNAPS checkpoint:

    ```python run_transductive_cnaps.py --data_path $RECORDS --feature_adaptation film --mode test -m ../model-checkpoints/meta-dataset-checkpoints/best_transductive_cnaps.pt --min_cluster_refinement_steps_test 2 --max_cluster_refinement_steps_test 4```

    The best performance was obtained at with minimum cluster refinement steps set to 2 and the maximum set to 4. You can of course try other max/min steps. Note that while the parameters are the same, since for testing, we sample a set of tasks from each dataset, variations may be seen in terms of reproducing results. That said, the discrepancies should be within the confidence intervals provided.
    
3. If you would like to train/test Transductive CNAPS from scratch, use the following two commands:
    
    ```python run_transductive_cnaps.py --data_path $RECORDS --feature_adaptation film --checkpoint_dir <address of the directory where you want to save the checkpoints> --min_cluster_refinement_steps_train 0 --max_cluster_refinement_steps_train 0 --min_cluster_refinement_steps_test 2 --max_cluster_refinement_steps_test 4```
    
    Note that we use a different min/max steps during training time (specifically both set to zero - no cluster refinements). To re-create results reported in the paper, please use ```--shuffle_dataset False```. To re-recreate our Meta-Dataset Leaderboard results (see leaderboard at https://github.com/google-research/meta-dataset#training-on-all-datasets), enable dataset shuffling via ```--shuffle_dataset True```. The ```--shuffle_dataset``` flag is set to ```True``` by default. Depending on your environment, training may take anywhere from 1-5 days. For reference, training on 2 T4 GPUs with 64G of memory and 8 dedicated GPUs took 2 days and 8 hours for Transductive CNAPS.
    
    If you are interested in training Transductive CNAPS with the Auto-Regressive Feature Extractor Adaptation Procedure (Transductive AR-CNAPS), you should theoretically be able to use the code provided, although we didn't run any experiments or test this variation directly.

## Meta-Dataset Results

| Dataset                         | Transductive CNAPS | Transductive CNAPS |
| ---                             | ---                | ---                |
| ```--shuffle_dataset```         | False              | True               |
| ---                             | ---                | ---                |
| In-Domain Datasets              | ---                | ---                |
| ILSVRC                          | 58.8±1.1           | 57.9±1.1           |
| Omniglot                        | 93.9±0.4           | 94.3±0.4           |
| Aircraft                        | 84.1±0.6           | 84.7±0.5           |
| Birds                           | 76.8±0.8           | 78.8±0.7           |
| Textures                        | 69.0±0.8           | 66.2±0.8           |
| Quick Draw                      | 78.6±0.7           | 77.9±0.6           |
| Fungi                           | 48.8±1.1           | 48.9±1.2           |
| VGG Flower                      | 91.6±0.4           | 92.3±0.4           |
| Out-of-Domain Datasets          | ---                | ---                |
| Traffic Signs                   | 76.1±0.7           | 59.7±1.1           |
| MSCOCO                          | 48.7±1.0           | 42.5±1.1           |
| MNIST                           | 95.7±0.3           | 94.7±0.3           |
| CIFAR10                         | 75.7±0.7           | 73.6±0.7           |
| CIFAR100                        | 62.9±1.0           | 61.8±1.0           |
| ---                             | ---                | ---                |
| In-Domain Average Accuracy      | 75.2±0.8           | 75.1±0.8           |
| Out-of-Domain Average Accuracy  | 71.8±0.8           | 66.5±0.8           |
| Overall Average Accuracy        | 73.9±0.8           | 71.8±0.8           |

## Mini/Tiered ImageNet Installations & Usage
In order to re-create these experiments, you need to:

1. First clone https://github.com/yaoyao-liu/mini-imagenet-tools, the mini-imagenet tools package used for generating tasks, and https://github.com/yaoyao-liu/tiered-imagenet-tools, the respective tiered-imagenet tools package under ```/transductive-cnaps-src```. Although theoretically this should sufficient, there may be errors arising from hard coded file paths (3 to 4 of which was present at the time of creating our set-up, although they seem to have been resolved since) which you can easily fix. Alternatively, we have included tested copies of both of these repositories within this director (see [miniimagenettools](https://github.com/plai-group/simple-cnaps/tree/master/transductive-cnaps-src/miniimagenettools/) and [tieredimagenettools](https://github.com/plai-group/simple-cnaps/tree/master/transductive-cnaps-src/tieredimagenettools/) which you can use to set up the respective datasets, should either repository be updated in the meantime).

2. Once the setup is complete, use ```run_transductive_cnaps_mt.py``` to run mini\tiered-imagenet experiments:
    
    ```cd src; python run_transductive_cnaps_mt.py --dataset <choose either mini or tiered> --feature_adaptation film --checkpoint_dir <address of the directory where you want to save the checkpoints> --pretrained_resnet_path <choose resnet pretrained checkpoint>```
    
**Note that as we emphasized this in the Simple CNAPS paper, Meta-Dataset pretrained CNAPS-based models including Simple CNAPS have a natural advantage on these benchmarks due to the pre-trianing of the feature extractor on the Meta-Dataset split of ImageNet. We alliviate this issue by re-training the ResNet feature extractor on the specific training splits of mini-ImageNet and tiered-ImageNet. These checkpoints have been provided under ```model-checkpoints/pretrained_resnets``` and are respectively ```pretrained_resnet_mini_imagenet.pt.tar``` and ```
pretrained_resnet_tiered_imagenet.pt.tar```. We additionally consider the case that additional non-test-set overlapping ImageNet classes are used to train our ResNet feature extractor in ```pretrained_resnet_mini_tiered_with_extra_classes.pt.tar```. We refer to this latter setup as "Feature Exctractor Trained (partially) on ImageNet", abbreviated as "FETI". Please visit the experimental section of our Transductive CNAPS paper for additional details on this setup.

## Mini-ImageNet Results

| Setup                           | 5-way 1-shot | 5-way 5-shot    | 10-way 1-shot    | 10-way 5-shot    |
| ---                             | ---          | ---             | ---              | ---              |
| Transductive CNAPS              | 55.6±0.9     | 73.1±0.7        | 42.8±0.7         | 59.6±0.5         |
| Transductive CNAPS + FETI       | 79.9±0.8     | 91.5±0.4        | 68.5±0.6         | 85.9±0.3         |

## Tiered-ImageNet Results

| Setup                           | 5-way 1-shot | 5-way 5-shot    | 10-way 1-shot    | 10-way 5-shot    |
| ---                             | ---          | ---             | ---              | ---              |
| Transductive CNAPS              | 65.9±1.0     | 81.8±0.7        | 54.6±0.8         | 72.5±0.6         |
| Transductive CNAPS + FETI       | 73.8±1.0     | 87.7±0.6        | 65.1±0.8         | 80.6±0.5         |

## Citation
We hope you have found this code base helpful! If you use this repository, please cite our papers:

```
@InProceedings{Bateni2020_SimpleCNAPS,
    author = {Bateni, Peyman and Goyal, Raghav and Masrani, Vaden and Wood, Frank and Sigal, Leonid},
    title = {Improved Few-Shot Visual Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}

@InProceedings{Bateni2022_TransductiveCNAPS,
    author    = {Bateni, Peyman and Barber, Jarred and van de Meent, Jan-Willem and Wood, Frank},
    title     = {Enhancing Few-Shot Image Classification With Unlabelled Examples},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {2796-2805}
}

@misc{Bateni2022_BeyondSimpleMetaLearning,
    title={Beyond Simple Meta-Learning: Multi-Purpose Models for Multi-Domain, Active and Continual Few-Shot Learning}, 
    author={Peyman Bateni and Jarred Barber and Raghav Goyal and Vaden Masrani and Jan-Willem van de Meent and Leonid Sigal and Frank Wood},
    year={2022},
    eprint={2201.05151},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
