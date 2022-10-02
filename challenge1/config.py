ENCODER = 'se_resnext50_32x4d'
# ENCODER = 'densenet161'
# ENCODER = 'efficientnet-b5'
ENCODER_WEIGHTS = 'imagenet'
# CLASSES = ['IMA']
# CLASSES = ['NA']
CLASSES = ['NE']
# CLASSES = ['NE_nohealth']
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
# ACTIVATION = None
DEVICE = 'cuda'
NAME = 'UNetpp_DICE'

# DATA_DIR = './data/A. Segmentation_IMA/'
DATA_DIR = './data/A. Segmentation_' + CLASSES[0] + '/'
