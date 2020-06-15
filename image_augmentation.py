import cv2
import pandas as pd
import os
import random
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
import itertools

seq = iaa.Sequential([
    iaa.Affine(rotate=(-5, 5), scale={"x": (0.3, 2), "y": (0.3, 2)}, ),
    iaa.Fliplr(0.5)
], random_order=True)

finalpath = ['D:\Aanshi\AANSHI2020\clique\data\img_augmented\PrivateTest\emotion0',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PrivateTest\emotion1',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PrivateTest\emotion2',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PrivateTest\emotion3',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PrivateTest\emotion4',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PrivateTest\emotion5',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PrivateTest\emotion6',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PublicTest\emotion0',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PublicTest\emotion1',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PublicTest\emotion2',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PublicTest\emotion3',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PublicTest\emotion4',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PublicTest\emotion5',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\PublicTest\emotion6',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\Training\emotion0',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\Training\emotion1',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\Training\emotion2',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\Training\emotion3',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\Training\emotion4',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\Training\emotion5',
             'D:\Aanshi\AANSHI2020\clique\data\img_augmented\Training\emotion6']

basepath = ['D:\Aanshi\AANSHI2020\clique\data\PrivateTest\emotion0',
            'D:\Aanshi\AANSHI2020\clique\data\PrivateTest\emotion1',
            'D:\Aanshi\AANSHI2020\clique\data\PrivateTest\emotion2',
            'D:\Aanshi\AANSHI2020\clique\data\PrivateTest\emotion3',
            'D:\Aanshi\AANSHI2020\clique\data\PrivateTest\emotion4',
            'D:\Aanshi\AANSHI2020\clique\data\PrivateTest\emotion5',
            'D:\Aanshi\AANSHI2020\clique\data\PrivateTest\emotion6',
            'D:\Aanshi\AANSHI2020\clique\data\PublicTest\emotion0',
            'D:\Aanshi\AANSHI2020\clique\data\PublicTest\emotion1',
            'D:\Aanshi\AANSHI2020\clique\data\PublicTest\emotion2',
            'D:\Aanshi\AANSHI2020\clique\data\PublicTest\emotion3',
            'D:\Aanshi\AANSHI2020\clique\data\PublicTest\emotion4',
            'D:\Aanshi\AANSHI2020\clique\data\PublicTest\emotion5',
            'D:\Aanshi\AANSHI2020\clique\data\PublicTest\emotion6',
            'D:\Aanshi\AANSHI2020\clique\data\Training\emotion0',
            'D:\Aanshi\AANSHI2020\clique\data\Training\emotion1',
            'D:\Aanshi\AANSHI2020\clique\data\Training\emotion2',
            'D:\Aanshi\AANSHI2020\clique\data\Training\emotion3',
            'D:\Aanshi\AANSHI2020\clique\data\Training\emotion4',
            'D:\Aanshi\AANSHI2020\clique\data\Training\emotion5',
            'D:\Aanshi\AANSHI2020\clique\data\Training\emotion6']

for path1,path2 in zip(basepath,finalpath):
    i=0
    entries = os.listdir(path1)
    for entry in os.listdir(path1):
        # Entry is only an image name. For reading via cv2, you need to give full path
        ##print("input img name:", entry)
        # To do that, you can use os.path.join() which will connect your imgname with your basepath
        img_full_path = os.path.join(path1, entry)
        ##print("Final img name:", img_full_path)
        # os.path.exists() to check our input path exists or not
        ##print(os.path.exists(img_full_path))
        img = cv2.imread(img_full_path)
        ##print(img.shape)
        image_aug = seq.augment_image(img)
        ##print('augmented ',i)
        img_path = "%s\img_%s.jpg" % (path2, i)
        ##print(image_aug.shape)
        ##print(img_path)
        i = i + 1
        cv2.imwrite(img_path, image_aug)
        ##print('success')

