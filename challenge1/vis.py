import os
import shutil

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn

# create test dataset
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.nn import functional as F

import config
from datasets import DRACDataset
from preprocess import get_validation_augmentation, get_preprocessing
from utils import visualize
from tqdm import tqdm

ENCODER = config.ENCODER
ENCODER_WEIGHTS = config.ENCODER_WEIGHTS
CLASSES = config.CLASSES
ACTIVATION = config.ACTIVATION
DEVICE = config.DEVICE
NAME = config.NAME

DATA_DIR = config.DATA_DIR

classes = {'IMA':'1. Intraretinal Microvascular Abnormalities', 'NA':'2. Nonperfusion Areas', 'NE':'3. Neovascularization', 'NE_nohealth':'3. Neovascularization'}

x_test_dir = os.path.join(DATA_DIR, '1. Original Images', 'b. Testing Set')
y_test_dir = os.path.join(DATA_DIR, '1. Original Images', 'b. Testing Set')
# x_test_dir = os.path.join(DATA_DIR, '1. Original Images', 'val')
# y_test_dir = os.path.join(DATA_DIR, '2. Groundtruths', 'val', classes[CLASSES[0]])
# y_test_dir = os.path.join(DATA_DIR, 'test', 'anno')
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

test_dataset = DRACDataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)

# test dataset without transformations for image visualization
test_dataset_vis = DRACDataset(
    x_test_dir, y_test_dir,
)

# load best saved checkpoint
best_model = torch.load('./models/'+CLASSES[0]+'_'+NAME+'_best_model.pth')

if not os.path.exists('./results'):
    os.makedirs('./results')
if not os.path.exists('./results/'+CLASSES[0]):
    os.makedirs('./results/'+CLASSES[0])
else:
    shutil.rmtree('./results/'+CLASSES[0])
    os.makedirs('./results/'+CLASSES[0])

print("product",CLASSES[0])
for i in tqdm(range(len(test_dataset))):
    n=i
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    gt_mask = test_dataset_vis[n][1].astype('uint8')
    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)

    pr_mask = (pr_mask.squeeze().cpu().numpy().round())


    # pr_mask[pr_mask>0] = 255
    cv2.imwrite('./results/'+CLASSES[0]+'/'+test_dataset_vis[n][2], pr_mask)
    # visualize(
    #     image=image_vis,
    #     ground_truth_mask=gt_mask,
    #     predicted_mask=pr_mask
    # )

if CLASSES[0] == 'NE_nohealth':
    if not os.path.exists('./results/NE'):
        shutil.copytree('./results/NE_nohealth', './results/NE')
    else:
        shutil.rmtree('./results/NE')
        shutil.copytree('./results/NE_nohealth', './results/NE')
    shutil.rmtree('./results/NE_nohealth')