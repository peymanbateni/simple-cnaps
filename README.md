# Improved Few-Shot Visual Classification

This repository contains source codes for the following papers:

- [Improved Few-Shot Visual Classification](https://openaccess.thecvf.com/content_CVPR_2020/html/Bateni_Improved_Few-Shot_Visual_Classification_CVPR_2020_paper.html) 
  
  @ IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020
  
- [Enhancing Few-Shot Image Classification with Unlabelled Examples](https://arxiv.org/abs/2006.12245) 

  @ IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022
  
- [Towards Better Few-Shot Object Recognition]() 
  
  @ IEEE TPAMI Special Issue on Learning with Fewer Labels in Computer Vision, 2022 (in submission)

The code base has been authored by Peyman Bateni, Jarred Barber, Raghav Goyal, Vaden Masrani, Dr. Jan-Willemn van de Meent, Dr. Leonid Sigal and Dr. Frank Wood. The source codes build on the original code base for CNAPS authored by Dr. John Bronskill, Jonathan Gordon, James Reqeima, Dr. Sebastian Nowozin, and Dr. Richard E. Turner. We would like to thank them for their help, support and early sharing of their work. To see the original CNAPS repository, visit https://github.com/cambridge-mlg/cnaps.

## Simple CNAPS

Simple CNAPS proposes the use of hierarchically regularized cluster means and covariance estimates within a Mahalanobis-distance based classifer for improved few-shot classification accuracy. This method incorporates said classifier within the same neural adaptive feature extractor as CNAPS. For more details, please refer to our paper on Simple CNAPS: [Improved Few-Shot Visual Classification](https://openaccess.thecvf.com/content_CVPR_2020/html/Bateni_Improved_Few-Shot_Visual_Classification_CVPR_2020_paper.html). The source code for this paper has been provided in the [simple-cnaps-src](https://github.com/plai-group/simple-cnaps/tree/master/simple-cnaps-src) directory. To reproduce our results, please refer to the README.md file within that folder.

Global Meta-Dataset Rank (Simple CNAPS): https://github.com/google-research/meta-dataset#training-on-all-datasets

Global Mini-ImageNet Rank (Simple CNAPS):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-mini-2)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-2?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-mini-3)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-3?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-mini-12)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-12?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-mini-13)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-13?p=improved-few-shot-visual-classification)

Global Tiered-ImageNet Rank (Simple CNAPS):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-tiered)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-tiered-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered-1?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-tiered-2)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered-2?p=improved-few-shot-visual-classification)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-few-shot-visual-classification/few-shot-image-classification-on-tiered-3)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered-3?p=improved-few-shot-visual-classification)

## Transductive CNAPS
Transductive CNAPS extends the Simple CNAPS framework to the transductive few-shot learning setting where all query examples are provided at once. This method uses a two-step transductive task-encoder for adapting the feature extractor as well as a soft k-means cluster refinement procedure, resulting in better test-time accuracy. For additional details, please refer to our paper on Transductive CNAPS: [Enhancing Few-Shot Image Classification with Unlabelled Examples](https://arxiv.org/abs/2006.12245). The source code for this work is provided under the [transductive-cnaps-src](https://github.com/plai-group/simple-cnaps/tree/master/transductive-cnaps-src) directory. To reproduce our results, please refer to the README.md file within this folder.

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

## Active and Continual Learning

We additionally evaluate both methods within the paradigms of "out of the box" active and continual learning. These settings were first proposed by Requeima et al., and studies how well few-shot classifiers, trained for few-shot learning, can be deployed for active and continual learning without any problem-specific finetuning or training. For additional details on our active and continual learning experiments and algorithms, please refer to our latest paper: [Towards Better Few-Shot Object Recognition](). For code and instructions to reproduce the experiments reported, please refer to the [active-learning](https://github.com/plai-group/simple-cnaps/tree/master/active-learning) and [continual-learning](https://github.com/plai-group/simple-cnaps/tree/master/continual-learning) folders.

## Meta-Dataset Results

| Dataset                         | Simple CNAPS | Simple CNAPS | Transductive CNAPS | Transductive CNAPS |
| ```--shuffle_dataset False```   | False        | True         | False              | True               |
| ---                             | ---          | ---          | ---                | ---                |
| In-Domain Datasets              | ---          | ---          | ---                | ---                |
| ILSVRC                          | 58.6±1.1     | 56.5±1.1     | 58.8±1.1           | 57.9±1.1           |
| Omniglot                        | 91.7±0.6     | 91.9±0.6     | 93.9±0.4           | 94.3±0.4           |
| Aircraft                        | 82.4±0.7     | 83.8±0.6     | 84.1±0.6           | 84.7±0.5           |
| Birds                           | 74.9±0.8     | 76.1±0.9     | 76.8±0.8           | 78.8±0.7           |
| Textures                        | 67.8±0.8     | 70.0±0.8     | 69.0±0.8           | 66.2±0.8           |
| Quick Draw                      | 77.7±0.7     | 78.3±0.7     | 78.6±0.7           | 77.9±0.6           |
| Fungi                           | 46.9±1.0     | 49.1±1.2     | 48.8±1.1           | 48.9±1.2           |
| VGG Flower                      | 90.7±0.5     | 91.3±0.6     | 91.6±0.4           | 92.3±0.4           |
| Out-of-Domain Datasets          | ---          | ---          | ---                | ---                |
| Traffic Signs                   | 73.5±0.7     | 59.2±1.0     | 76.1±0.7           | 59.7±1.1           |
| MSCOCO                          | 46.2±1.1     | 42.4±1.1     | 48.7±1.0           | 42.5±1.1           |
| MNIST                           | 93.9±0.4     | 94.3±0.4     | 95.7±0.3           | 94.7±0.3           |
| CIFAR10                         | 74.3±0.7     | 72.0±0.8     | 75.7±0.7           | 73.6±0.7           |
| CIFAR100                        | 60.5±1.0     | 60.9±1.1     | 62.9±1.0           | 61.8±1.0           |
| ---                             | ---          | ---          | ---                | ---                |
| In-Domain Average Accuracy      | 73.8±0.8     | 74.6±0.8     | 75.2±0.8           | 75.1±0.8           |
| Out-of-Domain Average Accuracy  | 69.7±0.8     | 65.8±0.8     | 71.8±0.8           | 66.5±0.8           |
| Overall Average Accuracy        | 72.2±0.8     | 71.2±0.8     | 73.9±0.8           | 71.8±0.8           |

## Mini-ImageNet Results

| Setup                           | 5-way 1-shot | 5-way 5-shot    | 10-way 1-shot    | 10-way 5-shot    |
| ---                             | ---          | ---             | ---              | ---              |
| Simple CNAPS                    | 53.2±0.9     | 70.8±0.7        | 37.1±0.5         | 56.7±0.5         |
| Transductive CNAPS              | 55.6±0.9     | 73.1±0.7        | 42.8±0.7         | 59.6±0.5         |
| ---                             | ---          | ---             | ---              | ---              |
| Simple CNAPS + FETI             | 77.4±0.8     | 90.3±0.4        | 63.5±0.6         | 83.1±0.4         |
| Transductive CNAPS + FETI       | 79.9±0.8     | 91.5±0.4        | 68.5±0.6         | 85.9±0.3         |

## Tiered-ImageNet Results

| Setup                           | 5-way 1-shot | 5-way 5-shot    | 10-way 1-shot    | 10-way 5-shot    |
| ---                             | ---          | ---             | ---              | ---              |
| Simple CNAPS                    | 63.0±1.0     | 80.0±0.8        | 48.1±0.7         | 70.2±0.6         |
| Transductive CNAPS              | 65.9±1.0     | 81.8±0.7        | 54.6±0.8         | 72.5±0.6         |
| ---                             | ---          | ---             | ---              | ---              |
| Simple CNAPS + FETI             | 71.4±1.0     | 86.0±0.6        | 57.1±0.7         | 78.5±0.5         |
| Transductive CNAPS + FETI       | 73.8±1.0     | 87.7±0.6        | 65.1±0.8         | 80.6±0.5         |

## Citation
We hope you have found our code base helpful! If you use this repository, please cite our papers:

```
@InProceedings{Bateni2020_SimpleCNAPS,
    author = {Bateni, Peyman and Goyal, Raghav and Masrani, Vaden and Wood, Frank and Sigal, Leonid},
    title = {Improved Few-Shot Visual Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}

@InProceedings{Bateni2020_TransductiveCNAPS,
    author    = {Peyman Bateni and Jarred Barber and Jan{-}Willem van de Meent and Frank Wood},
    title     = {Enhancing Few-Shot Image Classification with Unlabelled Examples},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
}

@article{Bateni2021_BetterFewShotObjectRecognition,
    author    = {Peyman Bateni and Jarred Barber and Raghav Goyal and Vaden Masrani and Jan{-}Willem van de Meent and Leonid Sigal and Frank Wood},
    title     = {Towards Better Few-Shot Object Recognition},
    year      = {2021}
}
```
