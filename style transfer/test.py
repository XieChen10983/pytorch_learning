# coding=gbk
"""
1. ��������
2. ����VGG19ģ��
3. ����������ʧ
4. ��������ʧ
5. ��VGGģ���п����ȡ��ʧ
6. �����Ż���
7. ѵ��������ͼƬ��ͼƬ�������ͼƬ���ƣ�����������ͼƬ����
"""
import torch
from torchvision import transforms
from PIL import Image

"""--------------------------------------------1. ��������-----------------------------------------------"""
# �̶�ͼƬ��С�����û��ʹ��GPU����һ����С�ߴ�
imsize = 512
is_cuda = torch.cuda.is_available()

# ת��ͼƬ��ʹ֮����VGGģ�͵�ѵ��
prep = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor(), transforms.Lambda(
    lambda x: x[torch.LongTensor([2, 1, 0])]
), transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)), ])
# �����ɵ�ͼƬת���ؿ��Գ��ֵĸ�ʽ
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                             ])
postpb = transforms.Compose([transforms.ToPILImage()])


# ȷ��ͼƬ���ݲ��ᳬ������ķ�Χ
def postp(tensor):
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img


# ʹ���ݼ��ظ��Ӽ򵥵Ĺ��ߺ���
def image_loader(image_name):
    image = Image.open(image_name)
    image = prep(image)
    # �����������ߴ�����ļ�������ߴ�
    image = image.unsqueeze(0)
    return image
