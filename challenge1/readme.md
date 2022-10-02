# Task 1 of DRAC Challenge
## Segmentation of DR lesions

### Install
First, install python by conda,
```
conda create -n segmentation_DRAC python=3.9 -y
conda activate segmentation_DRAC
```

Then, install environment by pip,
```
pip install -r requirements.txt
```

### Data Preparation
Mix the training dataset and the testing dataset download from [DRAC Challenge](https://drac22.grand-challenge.org/) into one package like the following format.
```
data
├── A. Segmentation
│   ├── 1. Original Images
│   │   ├── a. Training Set
│   │   └── b. Testing Set
│   └── 2. Groundtruths
└── spilt_val.py
```
If data is already, please run **spilt_val.py** like below command.
```
cd data
python spilt_val.py
```
Then, data preparation is over.
### Config
Before training or testing the model, please open **config.py** to change the config which you use during training and testing.
### Train
Just run **train.py** or **train.sh**, like below command,
```
python train.py
```
or
```
bash train.sh
```
### Test
Run **test.py**
### Others
**vis.py** is to product prediction in order to view or submit.


**ensemble.py** is to ensemble model and product prediction in order to submit

