#!/usr/bin/env python3
"""largely adapted from:
https://github.com/XuJiacong/PIDNet/blob/main/datasets/camvid.py
"""

import os
import os.path as osp
import json

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset
import matplotlib.pyplot as plt


class CamVid(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CamVid, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 11
        self.lb_ignore = 255
        self.color_list = [[0, 128, 192], [128, 0, 0], [64, 0, 128],
                             [192, 192, 128], [64, 64, 128], [64, 64, 0],
                             [128, 64, 128], [0, 0, 192], [192, 128, 128],
                             [128, 128, 128], [128, 128, 0]]

        self.lb_map = np.arange(256).astype(np.uint8)
        self.to_tensor = T.ToTensor(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2])*self.lb_ignore
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2)==3] = i
            # print(np.sum(label == i))
        # label = F.one_hot(label, self.n_cats).numpy()
        return label.astype(np.uint8)


    def get_image(self, impth, lbpth):
        # img = np.array(
        #     Image.open(impth).convert('RGB'))
        # label = np.array(
        #     Image.open(os.path.join(lbpth)).convert('BGR'))
        img = cv2.imread(impth)[:, :, ::-1].copy()
        label = cv2.imread(lbpth)[:, :, :].copy()
        # print(label)
        # print(img)

        label = self.color2label(label)
        print(np.min(label))
        return img, label


    # def __getitem__(self, idx):
    #     img, label = super(CamVid, self).__getitem__(idx)
    #     print(torch.unique(label))
    #     return img, label# F.one_hot(label, self.n_cats)
