# Improved Few-Shot Visual Classification

This directory contains the code for the paper, "Improved Few-Shot Visual Classification", which has been published at IEEE CVPR 2020. For a pdf copy of the paper, please visit IEEE CVF at https://openaccess.thecvf.com/content_CVPR_2020/html/Bateni_Improved_Few-Shot_Visual_Classification_CVPR_2020_paper.html or our ArXiv copy at https://arxiv.org/pdf/1912.03432.pdf.

Global Meta-Dataset Rank (Simple CNAPS): https://github.com/google-research/meta-dataset#training-on-all-datasets

Global mini-ImageNet Rank (Simple CNAPS):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-mini-2)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-2?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-mini-3)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-3?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-mini-12)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-12?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-mini-13)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-13?p=improved-few-shot-visual-classification)

Global tiered-ImageNet Rank (Simple CNAPS):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-tiered)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-tiered-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered-1?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-tiered-2)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered-2?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-tiered-3)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered-3?p=improved-few-shot-visual-classification)

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
    * Change to the ```simple-cnaps/simple-cnaps-src/``` directory in the repository.
    * Run: ```python prepare_extra_datasets.py```

## Meta-Dataset Usage
All test scripts used to produce results reported within the paper have been provided in the [test-scripts](https://github.com/plai-group/simple-cnaps/tree/master/simple-cnaps-src/test-scripts) directory. To train and test Simple CNAPs on Meta-Dataset:

1. First run the following three commands:
    
    ```ulimit -n 50000```
    
    ```export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>```
    
    ```export RECORDS=<root directory of where you have produced meta-dataset records>```
    
    Note that you may need to run the above commands every time you open a new command shell.
    
2. We have provided two checkpoints, correspondingly named "best_simple_cnaps.pt" and "best_simple_ar_cnaps.pt" under the [model-checkpoints](https://github.com/plai-group/simple-cnaps/tree/master/model-checkpoints/meta-dataset-checkpoints/) folder. These checkpoints contain the trained parameters for the two models that produced the results for Simple CNAPS and Simple AR-CNAPS (as referenced in the paper). To re-run evaluation, you can use the following commands to test the provided Simple CNAPS and Simple AR-CNAPS models:

    For Simple CNAPS:
    
    ```python run_simple_cnaps.py --data_path $RECORDS --feature_adaptation film --mode test -m ../model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt```
    
    For Simple AR-CNAPS:
    
    ```python run_simple_cnaps.py --data_path $RECORDS --feature_adaptation film+ar --mode test -m ../model-checkpoints/meta-dataset-checkpoints/best_simple_ar_cnaps.pt```

    Note that while the parameters are the same, since for testing, we sample a set of tasks from each dataset, minor variations may be seen in terms of reproducing results. That said, the discrepancies should be within the confidence intervals provided and should still match the referenced results considering statistical significance.
    
3. If you would like to train/test the models from scratch, use the following two commands:

    For Simple CNAPS:
    
    ```python run_simple_cnaps.py --data_path $RECORDS --feature_adaptation film --checkpoint_dir <address of the directory where you want to save the checkpoints>```
    
    For Simple AR-CNAPS:
    
    ```python run_simple_cnaps.py --data_path $RECORDS --feature_adaptation film+ar --checkpoint_dir <address of the directory where you want to save the checkpoints>```
    
    To re-create results reported in the paper, please use ```--shuffle_dataset False```. To re-recreate our Meta-Dataset Leaderboard results (see leaderboard at https://github.com/google-research/meta-dataset#training-on-all-datasets), enable dataset shuffling via ```--shuffle_dataset True```. The ```--shuffle_dataset``` flag is set to ```True``` by default. Depending on your environment, training may take anywhere from 1-5 days. For reference, training on 2 T4 GPUs with 64G of memory and 8 dedicated GPUs took 2 days and 4 hours for Simple CNAPS and over 3 days for Simple AR-CNAPS.

## Meta-Dataset Results

**Models trained on all datasets (with ```--shuffle_dataset False```)**

| Dataset                         | Simple CNAPS | Simple AR-CNAPS | CNAPS     | AR-CNAPS  |
| ---                             | ---          | ---             | ---       | ---       |
| In-Domain Datasets              | ---          | ---             | ---       | ---       |
| ILSVRC                          | 58.6±1.1     | 56.5±1.1        | 51.3±1.0  | 52.3±1.0  |
| Omniglot                        | 91.7±0.6     | 91.1±0.6        | 88.0±0.7  | 88.4±0.7  |
| Aircraft                        | 82.4±0.7     | 81.8±0.8        | 76.8±0.8  | 80.5±0.6  |
| Birds                           | 74.9±0.8     | 74.3±0.9        | 71.4±0.9  | 72.2±0.9  |
| Textures                        | 67.8±0.8     | 72.8±0.7        | 62.5±0.7  | 58.3±0.7  |
| Quick Draw                      | 77.7±0.7     | 75.2±0.8        | 71.9±0.8  | 72.5±0.8  |
| Fungi                           | 46.9±1.0     | 45.6±1.0        | 46.0±1.1  | 47.4±1.0  |
| VGG Flower                      | 90.7±0.5     | 90.3±0.5        | 89.2±0.5  | 86.0±0.5  |
| Out-of-Domain Datasets          | ---          | ---             | ---       | ---       |
| Traffic Signs                   | 73.5±0.7     | 74.7±0.7        | 60.1±0.9  | 60.2±0.9  |
| MSCOCO                          | 46.2±1.1     | 44.3±1.1        | 42.0±1.0  | 42.6±1.1  |
| MNIST                           | 93.9±0.4     | 95.7±0.3        | 88.6±0.5  | 92.7±0.4  |
| CIFAR10                         | 74.3±0.7     | 69.9±0.8        | 60.0±0.8  | 61.5±0.7  |
| CIFAR100                        | 60.5±1.0     | 53.6±1.0        | 48.1±1.0  | 50.1±1.0  |
| ---                             | ---          | ---             | ---       | ---       |
| In-Domain Average Accuracy      | 73.8±0.8     | 73.5±0.8        | 69.7±0.8  | 69.6±0.8  |
| Out-of-Domain Average Accuracy  | 69.7±0.8     | 67.6±0.8        | 61.5±0.8  | 59.8±0.8  |
| Overall Average Accuracy        | 72.2±0.8     | 71.2±0.8        | 66.5±0.8  | 65.9±0.8  |

**Models trained on all datasets (with ```--shuffle_dataset True```)**

| Dataset                         | Simple CNAPS | CNAPS         | 
| ---                             | ---          | ---           |
| In-Domain Datasets              | ---          | ---           |
| ILSVRC                          | 56.5±1.1     | 50.8±1.1      | 
| Omniglot                        | 91.9±0.6     | 91.7±0.5      | 
| Aircraft                        | 83.8±0.6     | 83.7±0.6      | 
| Birds                           | 76.1±0.9     | 73.6±0.9      | 
| Textures                        | 70.0±0.8     | 59.5±0.7      | 
| Quick Draw                      | 78.3±0.7     | 74.7±0.8      | 
| Fungi                           | 49.1±1.2     | 50.2±1.1      | 
| VGG Flower                      | 91.3±0.6     | 88.9±0.5      | 
| Out-of-Domain Datasets          | ---          | ---           |
| Traffic Signs                   | 59.2±1.0     | 56.5±1.1      |
| MSCOCO                          | 42.4±1.1     | 39.4±1.0      |
| MNIST                           | 94.3±0.4     | N/A           |
| CIFAR10                         | 72.0±0.8     | N/A           |
| CIFAR100                        | 60.9±1.1     | N/A           |
| ---                             | ---          | ---           |
| In-Domain Average Accuracy      | 74.6±0.8     | 71.6±0.8      |
| Out-of-Domain Average Accuracy  | 65.8±0.9     | 47.9±1.1*     |
| Overall Average Accuracy        | 71.2±0.8     | 66.9±0.9*     |

```*``` CNAPS averages don't include performances on MNIST, CIFAR10 and CIFAR100

## Mini/Tiered ImageNet Installations & Usage

In order to re-create these experiments, you need to:

1. First clone https://github.com/yaoyao-liu/mini-imagenet-tools, the mini-imagenet tools package used for generating tasks, and https://github.com/yaoyao-liu/tiered-imagenet-tools, the respective tiered-imagenet tools package under ```/simple-cnaps-src```. Although theoretically this should sufficient, there may be errors arising from hard coded file paths (3 to 4 of which was present at the time of creating our set-up, although they seem to have been resolved since) which you can easily fix. Alternatively, we have included tested copies of both of these repositories within this directory (see [miniimagenettools](https://github.com/plai-group/simple-cnaps/tree/master/simple-cnaps-src/miniimagenettools/) and [tieredimagenettools](https://github.com/plai-group/simple-cnaps/tree/master/simple-cnaps-src/tieredimagenettools/) which you can use to set up the respective datasets, should either repository be updated in the meantime).

2. Once the setup is complete, use ```run_simple_cnaps_mt.py``` to run mini\tiered-imagenet experiments:

   For Simple CNAPS:
    
    ```cd src; python run_simple_cnaps_mt.py --dataset <choose either mini or tiered> --feature_adaptation film --checkpoint_dir <address of the directory where you want to save the checkpoints> --pretrained_resnet_path <choose resnet pretrained checkpoint> --shot <task shot> --way <task way>```
    
   For Simple AR-CNAPS:
    
    ```cd src; python run_simple_cnaps_mt.py --dataset <choose either mini or tiered> --feature_adaptation film+ar --checkpoint_dir <address of the directory where you want to save the checkpoints> --pretrained_resnet_path <choose resnet pretrained checkpoint> --shot <task shot> --way <task way>```
    
**Note that as we emphasized this in the main paper, CNAPS-based models including Simple CNAPS have a natural advantage on these benchmarks due to the pre-trianing of the feature extractor on the Meta-Dataset split of ImageNet. We alliviate this issue by re-training the ResNet feature extractor on the specific training splits of mini-ImageNet and tiered-ImageNet. These checkpoints have been provided under [model-checkpoints/pretrained_resnets](https://github.com/plai-group/simple-cnaps/tree/master/model-checkpoints/pretrained_resnets/) and are respectively ```pretrained_resnet_mini_imagenet.pt.tar``` and ```
pretrained_resnet_tiered_imagenet.pt.tar```. We additionally consider the case where additional non-test-set overlapping ImageNet classes are used to train our ResNet feature extractor in ```pretrained_resnet_mini_tiered_with_extra_classes.pt.tar```. We refer to this latter setup as "Feature Exctractor Trained (partially) on ImageNet", abbreviated as "FETI". Please visit the experimental section of [Enhancing Few-Shot Image Classification with Unlabelled Examples](https://arxiv.org/abs/2006.12245) 
 for additional details on this setup.

**Updated results (with the new ResNet18 checkpoints - see explanation above) on mini-ImageNet**

| Setup                           | 5-way 1-shot | 5-way 5-shot    | 10-way 1-shot    | 10-way 5-shot    |
| ---                             | ---          | ---             | ---       | ---       |
| Simple CNAPS                    | 53.2±0.9     | 70.8±0.7        | 37.1±0.5  | 56.7±0.5  |
| Simple CNAPS + FETI             | 77.4±0.8     | 90.3±0.4        | 63.5±0.6  | 83.1±0.4  |

**Updated results (with the new ResNet18 checkpoints - see explanation above) on tiered-ImageNet**

| Setup                           | 5-way 1-shot | 5-way 5-shot    | 10-way 1-shot    | 10-way 5-shot    |
| ---                             | ---          | ---             | ---       | ---       |
| Simple CNAPS                    | 63.0±1.0     | 80.0±0.8        | 48.1±0.7  | 70.2±0.6  |
| Simple CNAPS + FETI             | 71.4±1.0     | 86.0±0.6        | 57.1±0.7  | 78.5±0.5  |

## Method Clarification - Use of 1/2 Coefficient on the Mahalanobis Distance

The 1/2 coefficient used in Equation 2, namely the one-half squared Mahalanobis distance, provides correpondence to Gaussian Mixture Models, but is widely believed to be of little significance as it corresponds to scaling vectors by a factor of 2. This scaling is naturally possible through a feature extractor and it was believed to not have a significant impact on the performance of the model. However, as we observed post-publication, it has a very small but nevertheless noticable impact as the use of the identity matrix in our regularization scheme breaks the equivalency previously believed. To this end, we provided a comparison in performance with respect to the presence of the 1/2 coefficient, and have updated the code to NOT use the coefficient as better performance is observed there. As shown, although overall performance changes are statistically insignificant, on certain specific datasets such as ILSVRC and VGG Flower, the difference is noticable. Note that this effect is primarily significant during training and at test time, either configuration achieves the same performance when used with the same checkpoints. The results shown below were generated with ```--shuffle_dataset False```.

| Dataset                         | Simple CNAPS (without 1/2) | Simple CNAPS (with 1/2)  |
| ---                             | ---                        | ---                      |
| In-Domain Datasets              | ---                        | ---                      |
| ILSVRC                          | 58.6±1.1                   | 55.3±1.1                 |
| Omniglot                        | 91.7±0.6                   | 90.7±0.6                 |
| Aircraft                        | 82.4±0.7                   | 82.0±0.7                 |
| Birds                           | 74.9±0.8                   | 74.2±0.9                 |
| Textures                        | 67.8±0.8                   | 66.0±0.8                 |
| Quick Draw                      | 77.7±0.7                   | 76.3±0.8                 |
| Fungi                           | 46.9±1.0                   | 47.3±1.0                 |
| VGG Flower                      | 90.7±0.5                   | 87.9±0.6                 |
| Out-of-Domain Datasets          | ---                        | ---                      |
| Traffic Signs                   | 73.5±0.7                   | 74.7±0.7                 |
| MSCOCO                          | 46.2±1.1                   | 47.4±1.1                 |
| MNIST                           | 93.9±0.4                   | 94.3±0.4                 |
| CIFAR10                         | 74.3±0.7                   | 72.0±0.8                 |
| CIFAR100                        | 60.5±1.0                   | 58.7±1.0                 |
| ---                             | ---                        | ---                      |
| In-Domain Average Accuracy      | 73.8±0.8                   | 72.5±0.8                 |
| Out-of-Domain Average Accuracy  | 69.7±0.8                   | 69.4±0.8                 |
| Overall Average Accuracy        | 72.2±0.8                   | 71.3±0.8                 |

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
