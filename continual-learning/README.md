# Beyond Simple Meta-Learning: Multi-Purpose Models for Multi-Domain, Active and Continual Few-Shot Learning - Continual Learning

This directory contains the code for the continual learning experiments referenced in the paper, [Beyond Simple Meta-Learning: Multi-Purpose Models for Multi-Domain, Active and Continual Few-Shot Learning](https://arxiv.org/abs/2201.05151), which is currently under review at the IEEE TPAMI Special Issue on Learning with Fewer Labels in Computer Vision 2022. We will release a PDF copy of this paper by mid-October.

We would like to also add that this code builds on code developed by Dr. John Bronskill, Jonathan Gordon, James Reqeima, Dr. Sebastian Nowozin, and Dr. Richard E. Turner and kindly shared by Dr. John Bronskil privately. As of writing, they have not publicly released their code for their continual learning experiments. If you find the continual learning code released here useful, we kingly ask that you also cite their relevant papers as noted in https://github.com/cambridge-mlg/cnaps. Our work was both inspired from their first proposal of "out of the box" few-shot classifiers for continual learning and was made possible thanks in part to the code shared with us early in our work.

## Dependencies
You can use the requirements.txt file included to install dependencies for both Simple CNAPS, Transductive CNAPS, relevant active/continual learning experiments and the accompanying source codes for Meta-Dataset, mini-ImageNet and tiered-ImageNet. To install all dependencies, ```run pip install -r requirements.txt```. In general, this code base requires Python 3.5 or greater, PyTorch 1.0 or greater and TensorFlow 2.0 or greater.

## GPU Requirements
We conducted our experiments on two Tesla P100 GPUs with about 10GB of memory each. Lower capacity GPUs can be used to run these experiments by using a lower batch size and modifying parts of the code, as long as the pre-trained checkpoints for Simple and Transductive CNAPS can fit on the GPU(s).

## Dataset Installation
We run active learning experiments on the MNIST, CIFAR100 and CIFAR10 benchmarks. For all datasets, the code to download the dataset has already been incorporated and the dataset will be setup during the first run. If you encounter any dataset issues, please make sure that the data paths defined within the load functions for each dataset in ```run_continual_learning.py``` are constructed. Depending on your setup, python is some times unable to create all directories needed.

## Running Continual Learning Experiments

All relavant test scripts used to produce the results reported in the paper have been added under the [test-scripts](https://github.com/plai-group/simple-cnaps/tree/master/continual-learning/test-scripts) folder in this directory. In general, to run a continual learning experiment, you can use the following command.

```
python -u run_continual_learning.py \
    --dataset <mnist, cifar10 or cifar100> \
    --test_shot <number of test examples per class> \
    --test_epochs <number of continual runs> \
    --model <choose model, simple_cnaps or transductive_cnaps> \
    --shot <number of labelled examples per class per task> \
    --head_type <"multi" uses a separate head for each task, "single" uses one single head for all tasks>
```

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

@phdthesis{Bateni2021_Thesis, 
    series      = {Electronic Theses and Dissertations (ETDs) 2008+}, 
    title       = {On label-eï¬€icient computer visionâ€¯: building fast and effective few-shot image classifiers}, 
    url         = {https://open.library.ubc.ca/collections/ubctheses/24/items/1.0402554}, 
    DOI         = {http://dx.doi.org/10.14288/1.0402554}, 
    school      = {University of British Columbia}, 
    author      = {Bateni, Peyman}, 
    year        = {2021}, 
    collection  = {Electronic Theses and Dissertations (ETDs) 2008+}
}
```
