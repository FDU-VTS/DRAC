import os

import torch
import numpy as np
import segmentation_models_pytorch as smp
import random
from torch.utils.data import DataLoader

import config
from datasets import DRACDataset
from preprocess import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils import AUC, Specificity,Dice


if __name__ == '__main__':
    random_seed = 1387
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    #cudnn.benchmark = True       
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ENCODER = config.ENCODER
    ENCODER_WEIGHTS = config.ENCODER_WEIGHTS
    CLASSES = config.CLASSES
    ACTIVATION = config.ACTIVATION
    DEVICE = config.DEVICE
    NAME = config.NAME

    # create segmentation model with pretrained encoder
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        in_channels=3,
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    DATA_DIR = config.DATA_DIR

    classes = {'IMA':'1. Intraretinal Microvascular Abnormalities', 'NA':'2. Nonperfusion Areas', 'NE':'3. Neovascularization', 'NE_nohealth':'3. Neovascularization'}
    x_train_dir = os.path.join(DATA_DIR, '1. Original Images', 'a. Training Set')
    y_train_dir = os.path.join(DATA_DIR, '2. Groundtruths', 'a. Training Set', classes[CLASSES[0]])

    x_valid_dir = os.path.join(DATA_DIR, '1. Original Images', 'val')
    y_valid_dir = os.path.join(DATA_DIR, '2. Groundtruths', 'val', classes[CLASSES[0]])

    train_dataset = DRACDataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = DRACDataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    # loss = smp.utils.losses.BCELoss()
    loss = smp.utils.losses.DiceLoss()
    # loss = FocalLoss()
    # loss = smp.losses.DiceLoss(mode='binary')
    # loss = smp.losses.SoftBCEWithLogitsLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        Dice(threshold=0.5)
        # smp.utils.metrics.Dice(threshold=0.5),
        # AUC(threshold=0.5),
        # smp.utils.metrics.Recall(threshold=0.5),
        # Specificity(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=1e-4), # origin 0.0001
    ])

    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.6)

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    if not os.path.exists('./models'):
        os.makedirs('./models')

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    log_txt = open('./logs/'+CLASSES[0]+'_'+NAME+'_best_model.txt', 'w')
    # train model for 100 epochs

    max_score = 0
    best_epoch = 0

    for i in range(0, 100):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        log_txt.write('\nEpoch: {} '.format(i))
        log_txt.write('iou_score: {} '.format(valid_logs['iou_score']))
        log_txt.write('dice_score: {} '.format(valid_logs['dice']))
        log_txt.flush()

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['dice']:
            max_score = valid_logs['dice']
        # if max_score < valid_logs['iou_score']:
        #     max_score = valid_logs['iou_score']
            best_epoch = i
            torch.save(model, './models/' + CLASSES[0] + '_'+NAME+'_best_model.pth')
            print('Model saved!')
        
        if CLASSES[0] != 'IMA':
            scheduler.step()
        else:
            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

    log_txt.write('\nBest epoch: {} '.format(best_epoch))
    log_txt.write('Best dice_score: {} '.format(max_score))
    log_txt.flush()
    log_txt.close()
