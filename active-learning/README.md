# Beyond Simple Meta-Learning: Multi-Purpose Models for Multi-Domain, Active and Continual Few-Shot Learning - Active Learning

This directory contains the code for the active learning experiments referenced in the paper, [Beyond Simple Meta-Learning: Multi-Purpose Models for Multi-Domain, Active and Continual Few-Shot Learning](https://arxiv.org/abs/2201.05151), which is currently under review at the Elsevier Neural Networks (Official Journal of the International Neural Network Society, European Neural Network Society & Japanese Neural Network Society) 2022.

We would like to also add that this code builds on code developed by Dr. John Bronskill, Jonathan Gordon, James Reqeima, Dr. Sebastian Nowozin, and Dr. Richard E. Turner and kindly shared by Dr. John Bronskil privately. As of writing, they have not publicly released their code for their active learning experiments. If you find the active learning code released here useful, we kingly ask that you also cite their relevant papers as noted in https://github.com/cambridge-mlg/cnaps. Our work was both inspired from their first proposal of "out of the box" few-shot classifiers for active learning and was made possible thanks in part to the code shared with us early in our work.

## Dependencies
You can use the requirements.txt file included to install dependencies for both Simple CNAPS, Transductive CNAPS, relevant active/continual learning experiments and the accompanying source codes for Meta-Dataset, mini-ImageNet and tiered-ImageNet. To install all dependencies, run ```pip install -r requirements.txt```. In general, this code base requires Python 3.5 or greater, PyTorch 1.0 or greater and TensorFlow 2.0 or greater.

## GPU Requirements
We conducted our experiments on two Tesla P100 GPUs with about 10GB of memory each. Lower capacity GPUs can be used to run these experiments by using a lower batch size and modifying parts of the code, as long as the pre-trained checkpoints for Simple and Transductive CNAPS can fit on the GPU(s).

## Dataset Installation
We run active learning experiments on the OMNIGLOT and CIFAR10 benchmarks. For CIFAR10, the code to download the dataset has already been incorporated and the dataset will be setup during the first run. For OMNIGLOT, however, you will need to download the evaluation split of the OMNIGLOT dataset from https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip, unzip and place all images inside an ```omniglot/test``` test folder inside your dataset folder. If you encounter any dataset issues, please make sure that the data paths defined within the load functions for each dataset in ```run_continual_learning.py``` are constructed. Depending on your setup, python is some times unable to create all directories needed.

## Running Active Learning Experiments

All relavant test scripts used to produce the results reported in the paper have been added under the [test-scripts](https://github.com/plai-group/simple-cnaps/tree/master/active-learning/test-scripts/) folder in this directory. In general, to run an active learning experiment, please run the following command.

```
python3 run_active_learning.py \
        --data_path <path to datasets, omniglot and cifar10> \
        --feature_adaptation <set to "film" but if your model uses a different strategy for adaptation, specify here> \
        --checkpoint_dir <path to checkpoint directory> \
        --model <specify model, one of simple_cnaps or transductive_cnaps> \
        --dataset <specify dataset name, one of omniglot or cifar 10>  \
        --test_model_path <path to trained model checkpoint> \
        --active_learning_method <active learning method to employ>
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

**If you would like to ask any questions or reach out regarding any of the papers, please email me directly at peyman.bateni@hotmail.com (my cs.ubc.ca email may have expired by the time you are emailing as I have graduated!).
