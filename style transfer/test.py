# coding=gbk
"""
1. 加载数据
2. 创建VGG19模型
3. 定义内容损失
4. 定义风格损失
5. 从VGG模型中跨层提取损失
6. 创建优化器
7. 训练并生成图片，图片风格与风格图片相似，内容与内容图片相似
"""
import torch
from torchvision import transforms
from PIL import Image

"""--------------------------------------------1. 加载数据-----------------------------------------------"""
# 固定图片大小，如果没有使用GPU，进一步减小尺寸
imsize = 512
is_cuda = torch.cuda.is_available()

# 转换图片，使之适于VGG模型的训练
prep = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor(), transforms.Lambda(
    lambda x: x[torch.LongTensor([2, 1, 0])]
), transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)), ])
# 将生成的图片转换回可以呈现的格式
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                             ])
postpb = transforms.Compose([transforms.ToPILImage()])


# 确保图片数据不会超出允许的范围
def postp(tensor):
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img


# 使数据加载更加简单的工具函数
def image_loader(image_name):
    image = Image.open(image_name)
    image = prep(image)
    # 拟合网络输入尺寸所需的假批处理尺寸
    image = image.unsqueeze(0)
    return image
