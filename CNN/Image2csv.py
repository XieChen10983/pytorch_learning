# coding=gbk
import numpy as np
import os
import csv
from time import time
from PIL import Image

Dir = "E:/machine learning/deep learning/pytorch/CNN/reshaped"

image_list = os.listdir(Dir)
image_path_list = [Dir + '/' + path for path in image_list]
print(image_path_list)
label_dict = {"cat": 0, "dog": 1}

# python2可以用file替代open
with open("E:/machine learning/deep learning/pytorch/CNN/catsanddogs.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)

    # # 先写入columns_name
    # writer.writerow(["index", "a_name", "b_name"])
    # # 写入多行用writerows
    # writer.writerows([np.array([0, 1, 3]), np.array([1, 2, 3]), np.array([2, 3, 4])])
    images = []
    for image_path in image_path_list[:2]:
        image = np.array(Image.open(image_path))
        r = image[:, :, 0].reshape((1, -1))
        g = image[:, :, 1].reshape((1, -1))
        b = image[:, :, 2].reshape((1, -1))
        row = np.concatenate((r, g, b), axis=1)
        print(row.shape)
        images.append(row)
    images = np.array(images, dtype=np.uint8).T.reshape((-1, 2))
    # print(images)
    writer.writerows(images)
