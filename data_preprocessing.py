# coding=gbk
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
from PIL import Image


class DogsAndCatsDataset(Dataset):
    def __init__(self, root_dir, size=(224, 224)):
        self.files = glob(root_dir)
        print(self.files)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        Img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        Label = self.files[idx].split('.')[-2]
        return Img, Label


if __name__ == '__main__':
    dogsset = DogsAndCatsDataset('E:/machine learning/datasets/猫狗检测/training_set/dogs/dog.*.jpg')
    # print(dogsset.__getitem__(1))
    epoch = 10
    for epo in range(1, epoch):
        print('-------------------------------', epo, '-------------------------------')
        dataloader = DataLoader(dogsset, batch_size=32, shuffle=True, num_workers=2)
        for batch, (imgs, labels) in enumerate(dataloader):
            # print(labels)
            # print('第%d循环， 第%d个批量。' % (epo, batch))
            print(imgs.size())
