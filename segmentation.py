'''

some scripts that could be useful, just to have them
'''

import os,cv2, labelme


import os, sys
path="images"
dirs = os.listdir(path)
i=0
for item in dirs:
   if item.endswith(".json"):
      if os.path.isfile(path+item):
         my_dest ="fin" + str(i)
         os.system("mkdir "+my_dest)
         os.system("labelme_json_to_dataset "+item+" -o "+my_dest)
         i=i+1

for item in os.listdir('images'):
    if item.endswith(".json"):
        labelme_json_to_dataset('images/' + item)
from glob import glob
import skimage, cv2
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
folders = glob("images2/*/")
i=0
for folder in folders:
    img = cv2.imread(folder+'img.png')
    mask = cv2.imread(folder+'label.png', cv2.IMREAD_GRAYSCALE)
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    _,mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite('dataset_2/images/'+ str(i) +'.png', img)
    cv2.imwrite('dataset_2/masks/' + str(i) +'.png', mask)
    i=i+1

from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=256 ,  input_height=1024, input_width=1024  )

model.train(
    train_images =  "dataset/images",
    train_annotations = "dataset/masks",

    checkpoints_path = "vgg_unet_1", epochs=15, steps_per_epoch=16, do_augment=True,augmentation_name="aug_all")
model.save('model_aug.h5')
