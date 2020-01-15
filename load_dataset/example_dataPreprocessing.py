# coding=gbk
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

mean = [123.68/255, 116.779/255, 103.939/255]
std = [58.393/255, 57.12/255, 57.375/255]
transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), (0.08, 1.0), ratio=(3./4, 4./3)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=(0.1, 0.1)),
    # transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std),
])

transform2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std),
])

imageSet = ImageFolder('./imagefolder', transform)
loader = DataLoader(imageSet, batch_size=1, shuffle=True)

import matplotlib.pyplot as plt
for image, label in loader:
    image = np.array(image)[0]
    # print(image)
    image = np.transpose(image, (1, 2, 0))
    image = (image*255).astype(np.uint8)
    plt.imshow(image)
    plt.title(str(label))
    plt.show()
