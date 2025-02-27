# coding=gbk

from PIL import Image
from torchvision import transforms

img_size = 512


def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img


def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()


# Img_path = './picture/content.png'
# Img = load_img(Img_path)
# show_img(Img)
