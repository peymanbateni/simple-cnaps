<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/tiered-imagenet-tools/master/tiered-imagenet.png" width="300"/>
</p>

# Tools for *tiered*ImageNet Dataset

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/y2l/tiered-imagenet-tools/blob/master/LICENSE)

This repo provides python source codes for creating *tiered*ImageNet dataset from ImageNet and the utils for generating batches during training. This repo is related to our work on few-shot learning: [Meta-Transfer Learning](https://github.com/y2l/meta-transfer-learning-tensorflow).

### Summary

* [About *tiered*ImageNet](#about-tieredImageNet)
* [Requirements](#requirements)
* [Usage](#usage)
* [Performance](#performance)
* [Acknowledgement](#acknowledgement)

### About *tiered*ImageNet

The [*tiered*ImageNet](https://arxiv.org/pdf/1803.00676.pdf) dataset is a larger subset of ILSVRC-12 with 608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated hierarchy. This set of nodes is partitioned into 20, 6, and 8 disjoint sets of training, validation, and testing nodes, and the corresponding classes form the respective meta-sets. As argued in Ren et al. (2018), this split near the root of the ImageNet hierarchy results in a more challenging, yet realistic regime with test classes that are less similar to training classes.

### Requirements

- Python 2.7 or 3.x
- numpy
- tqdm
- opencv-python
- Pillow

### Usage 
First, you need to download the image source files from [ImageNet website](http://www.image-net.org/challenges/LSVRC/2012/). If you already have it, you may use it directly.
```
Filename: ILSVRC2012_img_train.tar
Size: 138 GB
MD5: 1d675b47d978889d74fa0da5fadfb00e
```
Then clone the repo:
```
git clone https://github.com:y2l/tiered-imagenet-tools.git
cd tiered-imagenet-tools
```
To generate *tiered*ImageNet dataset from tar file:
```bash
python tiered_imagenet_generator.py --tar_dir [your_path_of_the_ILSVRC2012_img_train.tar]
```
To generate *tiered*ImageNet dataset from untarred folder:
```bash
python tiered_imagenet_generator.py --imagenet_dir [your_path_of_imagenet_folder]
```
If you want to resize the images to the specified resolution:
```bash
python tiered_imagenet_generator.py --tar_dir [your_path_of_the_ILSVRC2012_img_train.tar] --image_resize 100
```
P.S. In default settings, the images will be resized to 84 × 84. 

If you don't want to resize the images, you may set ```--image_resize 0```.

To use the ```TieredImageNetDataLoader``` class:
```python
from tiered_imagenet_dataloader import TieredImageNetDataLoader

dataloader = TieredImageNetDataLoader(shot_num=5, way_num=5, episode_test_sample_num=15)

dataloader.generate_data_list(phase='train')
dataloader.generate_data_list(phase='val')
dataloader.generate_data_list(phase='test')

dataloader.load_list(phase='all')

for idx in range(total_train_step):
    episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
        dataloader.get_batch(phase='train', idx=idx)
    ...
```

### Acknowledgement
This repo uses the source code from the following repos:

[Model-Agnostic Meta-Learning](https://github.com/cbfinn/maml)

[Optimization as a Model for Few-Shot Learning](https://github.com/gitabcworld/FewShotLearning)

