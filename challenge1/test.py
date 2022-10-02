import os

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

import config
from datasets import DRACDataset
from preprocess import get_validation_augmentation, get_preprocessing
from utils import Dice

if __name__ == '__main__':
    ENCODER = config.ENCODER
    ENCODER_WEIGHTS = config.ENCODER_WEIGHTS
    CLASSES = config.CLASSES
    ACTIVATION = config.ACTIVATION
    DEVICE = config.DEVICE
    NAME = config.NAME

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    DATA_DIR = config.DATA_DIR

    classes = {'IMA':'1. Intraretinal Microvascular Abnormalities', 'NA':'2. Nonperfusion Areas', 'NE':'3. Neovascularization', 'NE_nohealth':'3. Neovascularization'}

    x_test_dir = os.path.join(DATA_DIR, '1. Original Images', 'val')
    y_test_dir = os.path.join(DATA_DIR, '2. Groundtruths', 'val', classes[CLASSES[0]])

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        Dice(threshold=0.5)
        # smp.utils.metrics.Dice(threshold=0.5),
        # AUC(threshold=0.5),
        # smp.utils.metrics.Recall(threshold=0.5),
        # Specificity(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])
    # create test dataset
    test_dataset = DRACDataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset)

    # load best saved checkpoint
    best_model = torch.load('./models/' + CLASSES[0] + '_'+NAME+'_best_model.pth')

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)

    print(logs)