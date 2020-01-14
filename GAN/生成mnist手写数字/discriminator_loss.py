# coding=gbk
import torch
from torch.nn import BCEWithLogitsLoss


def discriminator_loss(Real_logits, Fake_logits, loss_function=BCEWithLogitsLoss()):
    """
    �����б�������ʧ������ÿһ����ͼ���б����������ͼ��������ͬ�����ķ�ֵ��
    ���ݷ�ֵ��ͼ��ԭ�������ʣ������٣������Լ������ʧ������
    :param Real_logits: ��ͼ�����εķ�����
    :param Fake_logits: ��ͼ�����εķ�����
    :param loss_function: ��ʧ������Ĭ��Ϊ��Ԫ�Ľ����غ�����
    :return: �ܵ���ʧ
    """
    # Batch size
    Real_Logits_num = Real_logits.size()
    Fake_Logits_num = Fake_logits.size()

    # Ŀ����True_logits��Untrue_logits,��ζ���б�����Ҫ��������ȷ��ȫ��ʶ��Ϊ��ȷ���������ȫ��ʶ��Ϊ����
    True_logits = torch.ones(Real_Logits_num).float()
    Untrue_logits = torch.zeros(Fake_Logits_num).float()

    # �������������ʧ
    Real_image_loss = loss_function(Real_logits, True_logits)  # ����ȷ��ʶ��Ϊ��ȷ�����Խ����ʧԽС
    Fake_image_loss = loss_function(Fake_logits, Untrue_logits)  # �������ʶ��Ϊ�������Խ����ʧԽС
    return Real_image_loss + Fake_image_loss


def discriminator_real_image_loss(Real_logits, loss_function=BCEWithLogitsLoss()):
    """
    �����б�������ʧ���б�ʱ�����ͼ��Ϊ��ͼ���б����������ͼ��������ͬ�����ķ�ֵ��
    ���ݷ�ֵ��ͼ��ԭ�������ʣ��棩�����Լ������ʧ������
    :param Real_logits: ��ͼ�����εķ�����
    :param loss_function: ��ʧ������Ĭ��Ϊ��Ԫ�Ľ����غ�����
    :return: ��ʧ
    """
    # Batch size
    Real_Logits_num = Real_logits.size()

    # Ŀ����True_logits��Untrue_logits,��ζ���б�����Ҫ��������ȷ��ȫ��ʶ��Ϊ��ȷ
    True_logits = torch.ones(Real_Logits_num).float()

    # �������������ʧ
    Real_image_loss = loss_function(Real_logits, True_logits)  # ����ȷ��ʶ��Ϊ��ȷ�����Խ����ʧԽС
    return Real_image_loss


def discriminator_fake_image_loss(Fake_logits, loss_function=BCEWithLogitsLoss()):
    """
    �����б�������ʧ���б�ʱ�����ͼ��Ϊ��ͼ���б����������ͼ��������ͬ�����ķ�ֵ��
    ���ݷ�ֵ��ͼ��ԭ�������ʣ��٣������Լ������ʧ������
    :param Fake_logits: ��ͼ�����εķ�����
    :param loss_function: ��ʧ������Ĭ��Ϊ��Ԫ�Ľ����غ�����
    :return: �ܵ���ʧ
    """
    # Batch size
    Fake_Logits_num = Fake_logits.size()

    # Ŀ����Untrue_logits,��ζ���б�����Ҫ�����������ȫ��ʶ��Ϊ����
    Untrue_logits = torch.zeros(Fake_Logits_num).float()

    # �������������ʧ
    Fake_image_loss = loss_function(Fake_logits, Untrue_logits)  # �������ʶ��Ϊ�������Խ����ʧԽС
    return Fake_image_loss
