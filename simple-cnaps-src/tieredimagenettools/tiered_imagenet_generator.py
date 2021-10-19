##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@u.nus.edu
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os
import numpy as np
import csv
from glob import glob
import cv2
import pdb
from shutil import copyfile
from tqdm import tqdm
from tqdm import trange

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--tar_dir',  type=str)
parser.add_argument('--imagenet_dir',  type=str)
parser.add_argument('--image_resize',  type=int,  default=84)

args = parser.parse_args()

class tieredImageNetGenerator(object):
    def __init__(self, input_args):
        self.input_args = input_args
        if self.input_args.tar_dir is not None:
            print('Untarring ILSVRC2012 package')
            self.imagenet_dir = './imagenet'
            if not os.path.exists(self.imagenet_dir):
                os.mkdir(self.imagenet_dir)
            os.system('tar xvf ' + str(self.input_args.tar_dir) + ' -C ' + self.imagenet_dir)
        elif self.input_args.imagenet_dir is not None:
            self.imagenet_dir = self.input_args.imagenet_dir
        else:
            print('You need to specify the ILSVRC2012 source file path')
        self.tiered_imagenet_dir = './tiered_imagenet'
        if not os.path.exists(self.tiered_imagenet_dir):
            os.mkdir(self.tiered_imagenet_dir)
        self.image_resize = self.input_args.image_resize

    def process_csv_files(self):
        self.train_class_list, _ = self.get_class_list('train')
        self.val_class_list, _ = self.get_class_list('val')
        self.test_class_list, _ = self.get_class_list('test')

        self.all_class_list = self.train_class_list + self.val_class_list + self.test_class_list

    def link_imagenet(self):
        images_keys = self.all_class_list
        target_base = self.tiered_imagenet_dir

        self.process_splits('train', target_base, self.imagenet_dir)
        self.process_splits('test', target_base, self.imagenet_dir)
        self.process_splits('val', target_base, self.imagenet_dir)

    def process_splits(self, split, target_base, img_dir):
        if split=='train':
            class_list = self.train_class_list
        elif split=='val':
            class_list = self.val_class_list
        elif split=='test':
            class_list = self.test_class_list
        else:
            print('Please set the correct class')     

        target_dir = target_base + '/' + split
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)        
   
        print('Process ' + split + ' data...')
        for i, keys in enumerate(class_list):
            this_class_target_dir = target_dir + '/' + keys + '/'
            if not os.path.exists(this_class_target_dir):
                os.mkdir(this_class_target_dir)  
            image_path = glob(os.path.join(img_dir, keys, '*'))
            for j in trange(len(image_path)):
                path = image_path[j]
                im = cv2.imread(path)
                im_resized = cv2.resize(im, (self.image_resize, self.image_resize), interpolation=cv2.INTER_AREA)
                cv2.imwrite(this_class_target_dir + keys + "%08d" % (j+1) + '.jpg', im_resized)
        
    def get_class_list(self, split='train'):
        filename = './tiered_imagenet_split/' + split + '.csv'
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            category_list = []
            class_list = []
            for row in csv_reader:
                category_list.append(row[1])
                class_list.append(row[0])
            category_list = list(set(category_list))
        return class_list, category_list     

if __name__ == "__main__":
    dataset_generator = tieredImageNetGenerator(args)
    dataset_generator.process_csv_files()
    dataset_generator.link_imagenet()

