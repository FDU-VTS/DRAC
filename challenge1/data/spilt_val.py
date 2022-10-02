import glob
import shutil
import os
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split



classes = {'IMA':'1. Intraretinal Microvascular Abnormalities', 'NA':'2. Nonperfusion Areas', 'NE':'3. Neovascularization', 'NE_nohealth':'3. Neovascularization'}
all_image = {'IMA':[[],[]],'NA':[[],[]],'NE':[[],[]],'NE_nohealth':[[],[]]}




# for key in classes.keys():
for key in classes:
    shutil.copytree('A. Segmentation','A. Segmentation_'+key)
    origin_image_train = './A. Segmentation_'+key+'/1. Original Images/a. Training Set'
    anno_imgae_train = './A. Segmentation_'+key+'/2. Groundtruths/a. Training Set'
    origin_image_val = './A. Segmentation_'+key+'/1. Original Images/val'
    anno_imgae_val = './A. Segmentation_'+key+'/2. Groundtruths/val'

    if not os.path.exists(anno_imgae_val):
        os.mkdir(anno_imgae_val)

    if not os.path.exists(origin_image_val):
        os.mkdir(origin_image_val)

    if not os.path.exists(os.path.join(anno_imgae_val,classes[key])):
        os.mkdir(os.path.join(anno_imgae_val,classes[key]))

    all_origin_image = os.listdir(os.path.join(anno_imgae_train,classes[key]))
    for image in os.listdir(origin_image_train):
        if image in all_origin_image:
            all_image[key][0].append(image)
        else:
            all_image[key][1].append(image)
    
    number = 2 if key != 'NE_nohealth' else 1 # donot process NE health
    for i in range(number):
        print(len(all_image[key][i]))
        train_set, val_set = train_test_split(all_image[key][i], test_size=0.2, random_state=42)
        for image in val_set:
            shutil.move(os.path.join(origin_image_train,image),os.path.join(origin_image_val,image))
            if i == 0:
                shutil.move(os.path.join(anno_imgae_train,classes[key],image),os.path.join(anno_imgae_val,classes[key],image))
            else:
                data = Image.open(os.path.join(origin_image_val,image))
                shape = data.size
                data.close()
                data = Image.new('L',(shape[0], shape[1]))
                data.save(os.path.join(anno_imgae_val,classes[key],image))
                # data.save('1.png')
        for image in train_set:
            if i == 1:
                data = Image.open(os.path.join(origin_image_train,image))
                shape = data.size
                data.close()
                data = Image.new('L',(shape[0], shape[1]))
                data.save(os.path.join(anno_imgae_train,classes[key],image))


 
if len(all_image['NE_nohealth'][0]) != 0 and len(all_image['NE_nohealth'][1]) != 0:
    for image in all_image['NE_nohealth'][1]:
        os.remove('A. Segmentation_NE_nohealth/1. Original Images/a. Training Set/'+image)
