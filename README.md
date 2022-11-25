# DRAC

This repo covers Team FDVTS_DR's solutions for MICCAI2022 Diabetic Retinopathy Analysis Challenge (DRAC).

## Dataset
We download the dataset from [DRAC2022](https://drac22.grand-challenge.org/Description/). For pre-training, we additionally adopt the [OCTA-25K-IQA-SEG](https://github.com/shanzha09/COIPS) dataset in challenge 2, and the [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection/) & [DDR](https://github.com/nkicsl/DDR-dataset) datasets in challenge 3.

## Task 1. segmentation of DR lesions
Please refer to [challenge1](challenge1) package


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

## Reference
If you use this code, please cite the following paper [**[pdf]**](https://arxiv.org/abs/2210.00515):
```
@article{hou2022deep,
  title={Deep-OCTA: Ensemble Deep Learning Approaches for Diabetic Retinopathy Analysis on OCTA Images},
  author={Hou, Junlin and Xiao, Fan and Xu, Jilan and Zhang, Yuejie and Zou, Haidong and Feng, Rui},
  journal={arXiv preprint arXiv:2210.00515},
  year={2022}
}
```
