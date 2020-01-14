# coding=gbk
import torch
from torch.nn import BCEWithLogitsLoss


def discriminator_loss(Real_logits, Fake_logits, loss_function=BCEWithLogitsLoss()):
    """
    计算判别器的损失。对于每一批的图像，判别器会给出和图像数量相同个数的分值。
    根据分值和图像原本的性质（即真或假），可以计算出损失函数。
    :param Real_logits: 真图像批次的分数。
    :param Fake_logits: 假图像批次的分数。
    :param loss_function: 损失函数，默认为二元的交叉熵函数。
    :return: 总的损失
    """
    # Batch size
    Real_Logits_num = Real_logits.size()
    Fake_Logits_num = Fake_logits.size()

    # 目标是True_logits和Untrue_logits,意味着判别器需要做到将正确的全部识别为正确，将错误的全部识别为错误
    True_logits = torch.ones(Real_Logits_num).float()
    Untrue_logits = torch.zeros(Fake_Logits_num).float()

    # 计算鉴别器的损失
    Real_image_loss = loss_function(Real_logits, True_logits)  # 将正确的识别为正确，相差越近损失越小
    Fake_image_loss = loss_function(Fake_logits, Untrue_logits)  # 将错误的识别为错误，相差越近损失越小
    return Real_image_loss + Fake_image_loss


def discriminator_real_image_loss(Real_logits, loss_function=BCEWithLogitsLoss()):
    """
    计算判别器的损失。判别时输入的图像为真图像，判别器会给出和图像数量相同个数的分值。
    根据分值和图像原本的性质（真），可以计算出损失函数。
    :param Real_logits: 真图像批次的分数。
    :param loss_function: 损失函数，默认为二元的交叉熵函数。
    :return: 损失
    """
    # Batch size
    Real_Logits_num = Real_logits.size()

    # 目标是True_logits和Untrue_logits,意味着判别器需要做到将正确的全部识别为正确
    True_logits = torch.ones(Real_Logits_num).float()

    # 计算鉴别器的损失
    Real_image_loss = loss_function(Real_logits, True_logits)  # 将正确的识别为正确，相差越近损失越小
    return Real_image_loss


def discriminator_fake_image_loss(Fake_logits, loss_function=BCEWithLogitsLoss()):
    """
    计算判别器的损失。判别时输入的图像为假图像，判别器会给出和图像数量相同个数的分值。
    根据分值和图像原本的性质（假），可以计算出损失函数。
    :param Fake_logits: 假图像批次的分数。
    :param loss_function: 损失函数，默认为二元的交叉熵函数。
    :return: 总的损失
    """
    # Batch size
    Fake_Logits_num = Fake_logits.size()

    # 目标是Untrue_logits,意味着判别器需要做到将错误的全部识别为错误
    Untrue_logits = torch.zeros(Fake_Logits_num).float()

    # 计算鉴别器的损失
    Fake_image_loss = loss_function(Fake_logits, Untrue_logits)  # 将错误的识别为错误，相差越近损失越小
    return Fake_image_loss
