# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import h5py
import numpy as np
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt

from pycocotools.coco import COCO

class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        #self.nJoints_coco = 17
        self.nJoints = 17

        #self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
        #                9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.accIdxs = range(1, self.nJoints+1)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

        # Load from coco
        coco = COCO("../data/coco/person_keypoints_train2017.json")
        self.imgname_coco_train = []
        self.bndbox_coco_train = []
        self.part_coco_train = []
        for ann in coco.dataset["annotations"]:
            img = coco.imgs[ann["image_id"]]
            kps = np.reshape(ann["keypoints"], (-1,3))[:,:2]
            self.imgname_coco_train.append(img["file_name"])
            self.bndbox_coco_train.append(ann["bbox"])
            self.part_coco_train.append(kps)
        coco = COCO("../data/coco/person_keypoints_val2017.json")
        self.imgname_coco_val = []
        self.bndbox_coco_val = []
        self.part_coco_val = []
        for ann in coco.dataset["annotations"]:
            img = coco.imgs[ann["image_id"]]
            kps = np.reshape(ann["keypoints"], (-1,3))[:,:2]
            self.imgname_coco_val.append(img["file_name"])
            self.bndbox_coco_val.append(ann["bbox"])
            self.part_coco_val.append(kps)

        self.size_train = len(self.imgname_coco_train)
        self.size_val = len(self.imgname_coco_val)

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]
        else:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]

        img_path = os.path.join(self.img_folder, imgname)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     'coco', sf, self, train=self.is_train)

        inp, out, setMask = metaData

        return inp, out, setMask, 'coco'

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
