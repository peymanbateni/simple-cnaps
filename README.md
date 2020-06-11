# Improved Few-Shot Visual Classification

This directory contains the code for the paper, "Improved Few-Shot Visual Classification", which has been published at IEEE CVPR 2020. For a pdf copy of the paper, please visit https://arxiv.org/pdf/1912.03432.pdf.

This code base builds upon the original source code for CNAPS authored by John Bronskill, Jonathan Gordon, James Reqeima, Sebastian Nowozin, and Richard E. Turner. As our code, proposing Simple CNAPS, extends the original code, we've included the corresponding license for the original source code. Furthermore, we've explicitly labelled our addition to CNAPS, using the commenting scheme """SCM: ...""" (standing for Simple CNAPS Modification) to denote our changes/extensions. To this end, unless denoted by such comments, all other code either belongs to the original CNAPS repository or contains rudimentary/semantic changes to the original course code.

To see the original CNAPS repository, visit https://github.com/cambridge-mlg/cnaps.

## Dependencies
This code requires the following:
* Python 3.5 or greater
* PyTorch 1.0 or greater
* TensorFlow 1.14 or greater (although we recommend refraining from using TensorFlow 2.0 since as of writing, the source code for the Meta-Dataset has not been updated to new TF version.)

## GPU Requirements
The GPU requirements for Simple CNAPS are:
* 2 GPUs with 16GB or more memory for training Simple AR-CNAPS
* 1 GPU with 16GB or more memory for training Simple CNAPS
We recommend the same settings for testing.

## Installation
Our installation process is the same as CNAPS:
1. Clone or download this repository.
2. Configure Meta-Dataset:
    * Note that as of writing, the Meta-Dataset repository is going through major updates. Should these updates cause any mismatches between our code and meta-dataset, we have included the version which we used in our work under ```meta-dataset-repo``` which can be used instead.
    * Otherwise, follow the the "User instructions" in the Meta-Dataset repository (https://github.com/google-research/meta-dataset)
    for "Installation" and "Downloading and converting datasets". This will take some time.
3. Install additional test datasets (MNIST, CIFAR10, CIFAR100):
    * Change to the $DATASRC directory: ```cd $DATASRC```
    * Download the MNIST test images: ```wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz```
    * Download the MNIST test labels: ```wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz```
    * Download the CIFAR10 dataset: ```wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz```
    * Extract the CIFAR10 dataset: ```tar -zxvf cifar-10-python.tar.gz```
    * Download the CIFAR100 dataset: ```wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz```
    * Extract the CIFAR10 dataset: ```tar -zxvf cifar-100-python.tar.gz```
    * Change to the ```simple-cnaps/src``` directory in the repository.
    * Run: ```python prepare_extra_datasets.py```

## Usage
To train and test Simple CNAPs on Meta-Dataset:

1. First run the following three commands:
    
    ```ulimit -n 50000```

    ```export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>```
    
    ```export RECORDS=<root directory of where you have produced meta-dataset records>```
    
    Note that you may need to run the above commands every time you open a new command shell.
    
2. We have provided two checkpoints, correspondingly named "best_simple_cnaps.pt" and "best_simple_ar_cnaps.pt". These checkpoints contain the trained parameters for the two models that produced the results for Simple CNAPS and Simple AR-CNAPS (as referenced in the paper). To re-run evaluation, you can use the following commands to test the provided Simple CNAPS and Simple AR-CNAPS models:

    For Simple CNAPS:
    
    ```cd src; python run_simple_cnaps.py --data_path $RECORDS --feature_adaptation film --mode test -m ../best_simple_cnaps.pt```
    
    For Simple AR-CNAPS:
    
    ```cd src; python src/run_simple_cnaps.py --data_path $RECORDS --feature_adaptation film+ar --mode test -m ../best_simple_ar_cnaps.pt```

    Note that while the parameters are the same, since for testing, we sample a set of tasks from each dataset, variations may be seen in terms of reproducing results. That said, the discrepancies should be within the confidence intervals provided and should still match the referenced results considering statistical significance.
    
3. If you would like to train/test the models from scratch, use the following two commands:

    For Simple CNAPS:
    
    ```cd src; python run_simple_cnaps.py --data_path $RECORDS --feature_adaptation film --checkpoint_dir <address of the directory where you want to save the checkpoints>```
    
    For Simple AR-CNAPS:
    
    ```cd src; python run_simple_cnaps.py --data_path $RECORDS --feature_adaptation film+ar --checkpoint_dir <address of the directory where you want to save the checkpoints>```
    
    Depending on your environment, training may take anywhere from 1-5 days. For reference, training on 2 T4 GPUs with 64G of memory and 8 dedicated GPUs took 2 days and 4 hours for Simple CNAPS and over 3 days for Simple AR-CNAPS.

**Models trained on all datasets**

| Dataset       | Simple CNAPS | Simple AR-CNAPS | CNAPS     | AR-CNAPS |
| ---           | --- | ---           | ---      | ---       |
| In-Domain Datasets           | --- | ---           | ---      | ---       |
| ILSVRC        | 58.6±1.1 | 56.5±1.1      | 51.3±1.0 | 52.3±1.0  |
| Omniglot      | 91.7±0.6 | 91.1±0.6      | 88.0±0.7 | 88.4±0.7  |
| Aircraft      | 82.4±0.7 | 81.8±0.8      | 76.8±0.8 | 80.5±0.6  |
| Birds         | 74.9±0.8 | 74.3±0.9      | 71.4±0.9 | 72.2±0.9  |
| Textures      | 67.8±0.8 | 72.8±0.7      | 62.5±0.7 | 58.3±0.7  |
| Quick Draw    | 77.7±0.7 | 75.2±0.8      | 71.9±0.8 | 72.5±0.8  |
| Fungi         | 46.9±1.0 | 45.6±1.0      | 46.0±1.1 | 47.4±1.0  |
| VGG Flower    | 90.7±0.5 | 90.3±0.5      | 89.2±0.5 | 86.0±0.5  |
| Out-of-Domain Datasets           | --- | ---           | ---      | ---       |
| Traffic Signs | 73.5±0.7 | 74.7±0.7      | 60.1±0.9 | 60.2±0.9  |
| MSCOCO        | 46.2±1.1 | 44.3±1.1      | 42.0±1.0 | 42.6±1.1  |
| MNIST         | 93.9±0.4 | 95.7±0.3      | 88.6±0.5 | 92.7±0.4  |
| CIFAR10       | 74.3±0.7 | 69.9±0.8      | 60.0±0.8 | 61.5±0.7  |
| CIFAR100      | 60.5±1.0 | 53.6±1.0      | 48.1±1.0 | 50.1±1.0  |
| ---           | --- | ---           | ---      | ---       |
| In-Domain Average Accuracy  | 73.8±0.8 | 73.5±0.8           | 69.7±0.8      | 69.6±0.8       |
| Out-of-Domain Average Accuracy  | 69.7±0.8 | 67.6±0.8           | 61.5±0.8      | 59.8±0.8       |
| Overall Average Accuracy  | 72.2±0.8 | 71.2±0.8           | 66.5±0.8      | 65.9±0.8       |


## Method Clarrification - Use of 1/2 Coefficient on the Mahalanobis Distance

The 1/2 coefficient used in Equation 2, namely the one-half squared Mahalanobis distance, provides correpondence to Gaussian Mixutre Models, but is widely believed to be of little significance as it corresponds to scaling vectors by a factor of 2. This scaling is naturally possible through a feature extractor and it was believed to not have a significant impact on the performance of the model. However, as we observed post-publication, it has a very small but nevertheless noticable impact as the use of the identity matrix in our regularization scheme breaks the equivalency previously believed. To this end, we provided a comparison in performance with respect to the presence of the 1/2 coefficient, and have updated the code to NOT use the coefficient as better performance is observed there. As shown, although overall performance changes are statistically insignificant, on certain specific datasets such as ILSVRC and VGG Flower, the difference is noticable. Note that this effect is primarily significant during training and at test time, either configuration achieves the same performance when used with the same checkpoints.

| Dataset       | Simple CNAPS (without 1/2) | Simple CNAPS (with 1/2)  |
| ---           | --- | ---           |
| In-Domain Datasets           | --- | ---           |
| ILSVRC        | 58.6±1.1 | 55.3±1.1      |
| Omniglot      | 91.7±0.6 | 90.7±0.6      |
| Aircraft      | 82.4±0.7 | 82.0±0.7      |
| Birds         | 74.9±0.8 | 74.2±0.9      |
| Textures      | 67.8±0.8 | 66.0±0.8      |
| Quick Draw    | 77.7±0.7 | 76.3±0.8      |
| Fungi         | 46.9±1.0 | 47.3±1.0      |
| VGG Flower    | 90.7±0.5 | 87.9±0.6      |
| Out-of-Domain Datasets           | --- | ---           |
| Traffic Signs | 73.5±0.7 | 74.7±0.7      |
| MSCOCO        | 46.2±1.1 | 47.4±1.1      |
| MNIST         | 93.9±0.4 | 94.3±0.4      |
| CIFAR10       | 74.3±0.7 | 72.0±0.8      |
| CIFAR100      | 60.5±1.0 | 58.7±1.0      |
| ---           | --- | ---           |
| In-Domain Average Accuracy  | 73.8±0.8 | 72.5±0.8           |
| Out-of-Domain Average Accuracy  | 69.7±0.8 | 69.4±0.8           |
| Overall Average Accuracy  | 72.2±0.8 | 71.3±0.8           |

## Citing this repository/paper
```
@misc{bateni2019improved,
    title={Improved Few-Shot Visual Classification},
    author={Peyman Bateni and Raghav Goyal and Vaden Masrani and Frank Wood and Leonid Sigal},
    year={2019},
    eprint={1912.03432},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
