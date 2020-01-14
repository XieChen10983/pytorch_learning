# coding=gbk
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from random import shuffle

in_path = 'E:\\machine learning\\datasets\\猫狗检测全\\train'
out_path = 'E:/machine learning/deep learning/pytorch/CNN/reshaped28_28/'
name_list = os.listdir(in_path)
name_list = [in_path+'\\'+name for name in name_list]
# print(name_list)
# name_list = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                       [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                       [2, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                       [3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                       [4, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                       [5, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# np.random.shuffle(name_list)
# print(name_list)
# print('正在保存图像...')
num_sum = len(name_list)
shuffle(name_list)
for num, name in enumerate(name_list):
    img = Image.open(name).resize((28, 28))
    # print(img.size)
    # Img = np.array(img)
    # print(Img.shape)
    # plt.imshow(Img)
    # plt.show()
    save_path = out_path+name.split('\\')[-1]
    img.save(save_path)
    if num % 500 == 0:
        print('['+'='*2*int(num/500)+'>'+'%f' % (num/num_sum*100)+'%'+']')
print('图像保存结束！')
