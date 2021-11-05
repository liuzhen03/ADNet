#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: dataset.py 
@time: 2021/08/02
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm 

"""
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import os.path as osp
from utils.utils import *
import cv2


class NTIRE_Training_Dataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = osp.join(root_dir, 'training')
        self.scenes_dir_list = sorted(os.listdir(self.root_dir))
        self.image_list = []
        for scene in range(len(self.scenes_dir_list)):
            exposures_path = osp.join(self.root_dir, self.scenes_dir_list[scene], 'exposures.npy')
            align_ratio_path = osp.join(self.root_dir, self.scenes_dir_list[scene], 'alignratio.npy')
            image_path = sorted(glob.glob(os.path.join(self.root_dir, self.scenes_dir_list[scene], '*.png')))
            self.image_list += [[exposures_path, align_ratio_path, image_path]]

    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = np.load(self.image_list[index][0])
        floating_exposures = exposures - exposures[1]
        # Read LDR images
        ldr_images = ReadImages(self.image_list[index][2][:-1])
        # Read HDR label
        label = imread_uint16_png(self.image_list[index][2][-1], self.image_list[index][1])
        # ldr images process
        s_gamma = 2.24
        if random.random() < 0.3:
            s_gamma += (random.random() * 0.2 - 0.1)
        image_short = ev_alignment(ldr_images[0], floating_exposures[0], s_gamma)
        # image_medium = ev_alignment(ldr_images[1], floating_exposures[1], 2.24)
        image_medium = ldr_images[1]
        image_long = ev_alignment(ldr_images[2], floating_exposures[2], s_gamma)

        image_short_concat = np.concatenate((ldr_images[0], image_short), 2)
        image_medium_concat = np.concatenate((ldr_images[1], image_medium), 2)
        image_long_concat = np.concatenate((ldr_images[2], image_long), 2)

        img0 = image_short_concat.astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium_concat.astype(np.float32).transpose(2, 0, 1)
        img2 = image_long_concat.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label}
        return sample

    def __len__(self):
        return len(self.scenes_dir_list)


class NTIRE_Validation_Dataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = osp.join(root_dir, 'validation')
        self.image_ids = np.load(os.path.join(root_dir, 'val.npy')).tolist()
        self.crop_size = 800
        self.image_list = []
        for image_id in self.image_ids:
            exposures_path = osp.join(self.root_dir, "{:04d}_exposures.npy".format(image_id))
            align_ratio_path = osp.join(self.root_dir, "{:04d}_alignratio.npy".format(image_id))
            image_short_path = os.path.join(self.root_dir, "{:04d}_short.png".format(image_id))
            image_medium_path = os.path.join(self.root_dir, "{:04d}_medium.png".format(image_id))
            image_long_path = os.path.join(self.root_dir, "{:04d}_long.png".format(image_id))
            image_gt_path = os.path.join(self.root_dir, "{:04d}_gt.png".format(image_id))
            image_path = [image_short_path, image_medium_path, image_long_path, image_gt_path]
            self.image_list += [[exposures_path, align_ratio_path, image_path]]

    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = np.load(self.image_list[index][0])
        floating_exposures = exposures - exposures[1]
        image_name = self.image_list[index][2][-1].split('/')[-1][:4]
        # Read LDR images
        ldr_images = ReadImages(self.image_list[index][2][:-1])
        # Read HDR label
        label = imread_uint16_png(self.image_list[index][2][-1], self.image_list[index][1])
        # ldr images process
        image_short = ev_alignment(ldr_images[0], floating_exposures[0], 2.24)
        image_medium = ldr_images[1]
        image_long = ev_alignment(ldr_images[2], floating_exposures[2], 2.24)
        image_short_concat = np.concatenate((image_short, ldr_images[0]), 2)
        image_medium_concat = np.concatenate((image_medium, ldr_images[1]), 2)
        image_long_concat = np.concatenate((image_long, ldr_images[2]), 2)

        x = 0
        y = 0
        img0 = image_short_concat[x:x + self.crop_size, y:y + self.crop_size ].astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium_concat[x:x + self.crop_size, y:y + self.crop_size ].astype(np.float32).transpose(2, 0, 1)
        img2 = image_long_concat[x:x + self.crop_size, y:y + self.crop_size ].astype(np.float32).transpose(2, 0, 1)
        label = label[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label, 'image_name': image_name}
        return sample

    def __len__(self):
        return len(self.image_ids)