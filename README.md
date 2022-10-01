# DRAC

This repo covers Team FDVTS_DR's solutions for MICCAI2022 Diabetic Retinopathy Analysis Challenge (DRAC).

## Dataset
We download the dataset from [DRAC2022](https://drac22.grand-challenge.org/Description/). For pre-training, we additionally adopt the [OCTA-25K-IQA-SEG](https://github.com/shanzha09/COIPS) dataset in challenge 2, and the [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection/) & [DDR](https://github.com/nkicsl/DDR-dataset) datasets in challenge 3.

## Task 1. segmentation of DR lesions


## Task 2. image quality assessment
```
cd challenge2&3
```
**train the OCTA-25K-IQA-SEG pre-trained vit-s model with mixup and cutmix**
```
python main.py --challenge 2 --model vit --KK 0 [--pretrained True]  [--mixup True]  --visname 2_vit_mix_cut_KK0_pre 
```

## Task 3. DR grading
```
cd challenge2&3
```
**train the EyePACS & DDR pre-trained vit-s model with mixup and cutmix**
```
python main.py --challenge 3 --model vit --KK 0 [--pretrained True]  [--mixup True]  --visname 3_vit_mix_cut_KK0_pre 
```

