import os
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T 
from PIL import Image
import torch
import csv
from random import shuffle, sample
from numpy.random import choice
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')

import openpyxl as pxl
import cv2
import pandas as pd
from sklearn.model_selection import KFold
from skimage import img_as_ubyte
from torchvision import utils as vutils
import time
from scipy import ndimage



class quality_dataset(data.Dataset):
    def __init__(self, train=False, val=False, test=False, test_tta=False, all=False, KK=0):
        self.train = train
        self.val = val
        self.test = test
        self.path = 'data/B. Image Quality Assessment/'

        if train or val or all:
            self.file = '1. Original Images/a. Training Set/'
            e_file = '2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv'
        else:
            self.file = 'data/DRAC2022_Testing_Set/B. Image Quality Assessment/1. Original Images/b. Testing Set/'
        self.imgs = []
        img_list = [[] for _ in range(3)]

        if test or test_tta:
            for i in range(len(os.listdir(self.file))):
                x = os.listdir(self.file)[i]
                self.imgs.append([self.file+x, -1, x])

        #     # OCTA-25K-IQA-SEG
        #     # path = 'data/quality_assessment/3x3/For_Train/test/'
        #     # for classes in range(3):
        #     #     for image in os.listdir(path+str(classes)):
        #     #         self.imgs.append([path+str(classes)+'/'+image, classes, image])
  

        elif train or val:
            csv_file = pd.read_csv(self.path + e_file)
            self.dict_label = {}
            for index, row in csv_file.iterrows():
                image_id = row['image name']
                rank = int(row['image quality level'])
                img_list[rank].append(image_id)

            for i in range(3):
                # print("CV:",KK)
                kf = KFold(n_splits=5,shuffle=True,random_state=5)
                for kk, (a,b) in enumerate(kf.split(range(len(img_list[i])))):
                    if kk == KK:
                        train_index, val_index = a, b
                        if self.train:
                            print("Grade",i, ':',  len(train_index),len(val_index))
                if train:
                    for index in train_index:
                        x = img_list[i][index]
                        self.imgs.append([self.path+self.file+x, i, x])
                else:
                    for index in val_index:
                        x = img_list[i][index]
                        self.imgs.append([self.path+self.file+x, i, x])       
        elif all:
            csv_file = pd.read_csv(self.path + e_file)
            for index, row in csv_file.iterrows():
                image_id = row['image name']
                rank = int(row['image quality level'])
                self.imgs.append([self.path+self.file+image_id, rank, image_id])       

        '''
        # OCTA-25K-IQA-SEG
        self.imgs = []
        if train:
            path = 'data/quality_assessment/6x6/For_Train/train/'
            for rank in range(3):
                img_path = path + str(rank) + '/'
                for image_id in os.listdir(img_path):
                    self.imgs.append([img_path+image_id, rank, image_id])
        elif val:
            path = 'data/quality_assessment/6x6/For_Train/test/'
            for rank in range(3):
                img_path = path + str(rank) + '/'
                for image_id in os.listdir(img_path):
                    self.imgs.append([img_path+image_id, rank, image_id])
        '''
        data_aug = {
            'brightness': 0.4,  # how much to jitter brightness
            'contrast': 0.4,  # How much to jitter contrast
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            'img_size': 384
        }
        if train:
            self.transform = T.Compose([
                T.Resize((640,640)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop(
                    size=((data_aug['img_size'], data_aug['img_size'])),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                ),

                T.ColorJitter(
                    brightness=data_aug['brightness'],
                    contrast=data_aug['contrast'],
                ),
                T.ToTensor(),
                # T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]) # copis pretrained
            ])

           
        elif val or test or all:
            self.transform = T.Compose([
                T.Resize((data_aug['img_size'],data_aug['img_size'])),
                T.ToTensor(),
                # T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
            ])

        elif test_tta:
            self.transform = T.Compose([
                T.Resize((640,640)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),  
                T.RandomResizedCrop(
                    size=((data_aug['img_size'], data_aug['img_size'])),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                ),
                T.ToTensor(),
                # T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]) # copis pretrained
            ])            
        print(len(self.imgs))
        
    def __getitem__(self, index):
        img, label, name = self.imgs[index]
        data = Image.open(img).convert('RGB')
        data = self.transform(data)
        return data, label, name

    def __len__(self):
        return len(self.imgs)



class grading_dataset(data.Dataset):
    def __init__(self, train=False, val=False, test=False, test_tta=False, all=False, KK=0):
        self.train = train
        self.val = val
        self.test = test
        self.path = 'data/C. Diabetic Retinopathy Grading/'

        if train or val or all:
            self.file = '1. Original Images/a. Training Set/'
            e_file = '2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv'
        else:
            self.file = '/home/feng/hjl/DRAC/data/DRAC2022_Testing_Set/C. Diabetic Retinopathy Grading/1. Original Images/b. Testing Set/'
        self.imgs = []
        img_list = [[] for _ in range(3)]

        if test or test_tta:
            for i in range(len(os.listdir(self.file))):
                x = os.listdir(self.file)[i]
                self.imgs.append([self.file+x, -1, x])

        elif train or val:
            csv_file = pd.read_csv(self.path + e_file)
            self.dict_label = {}
            for index, row in csv_file.iterrows():
                image_id = row['image name']
                rank = int(row['DR grade'])
                img_list[rank].append(image_id)

            classes = [0,1,2]
            for i in classes:
                # print("CV:",KK)
                kf = KFold(n_splits=5,shuffle=True,random_state=5)
                for kk, (a,b) in enumerate(kf.split(range(len(img_list[i])))):
                    if kk == KK:
                        train_index, val_index = a, b
                        if self.train:
                            print("Grade",i, ':',  len(train_index),len(val_index))
                if train:
                    for index in train_index:
                        x = img_list[i][index]
                        self.imgs.append([self.path+self.file+x, i, x])
                else:
                    for index in val_index:
                        x = img_list[i][index]
                        self.imgs.append([self.path+self.file+x, i, x])       
        elif all:
            csv_file = pd.read_csv(self.path + e_file)
            self.dict_label = {}
            for index, row in csv_file.iterrows():
                image_id = row['image name']
                rank = int(row['DR grade'])
                self.imgs.append([self.path+self.file+image_id, rank, image_id])       
       

        data_aug = {
            'brightness': 0.4,  # how much to jitter brightness # 0.8,1.2
            'contrast': 0.4,  # How much to jitter contrast
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            'degrees': (-180, 180),  # range of degrees to select from # vit:-180
            'img_size': 384
        }
        if train:
            self.transform = T.Compose([
                T.Resize((640,640)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(
                    brightness=data_aug['brightness'],
                    contrast=data_aug['contrast'],
                ),
                T.RandomResizedCrop(
                    size=((data_aug['img_size'], data_aug['img_size'])),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                ),
                T.RandomAffine(
                    degrees=data_aug['degrees'],
                ),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ])

           
        elif val or test or all:
            self.transform = T.Compose([
                T.Resize((data_aug['img_size'],data_aug['img_size'])),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ])


        elif test_tta:
            self.transform = T.Compose([
                T.Resize((data_aug['img_size'],data_aug['img_size'])),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                ),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ])

        print(len(self.imgs))
        
    def __getitem__(self, index):
        img, label, name = self.imgs[index]
        data = Image.open(img).convert('RGB')
        data = self.transform(data)

        return data, label, name

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dst = grading_dataset(train=True)
    for i in range(dst.__len__()):
        img, label, name = dst.__getitem__(i)
        if i == 10:
            break
