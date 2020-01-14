# coding=gbk
from torch import nn, optim
from build_model import get_style_model_and_loss


def get_input_param_optimizer(input_img):
    """
    input_img is a variable.
    :param input_img:
    :return:
    """
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


def run_style_transfer(content_img, style_img, input_img, num_epochs=300):
    print('Building the style transfer model...')
    model, style_loss_list, content_loss_list = get_style_model_and_loss(style_img, content_img)
    input_param, optimizer = get_input_param_optimizer(input_img)
    # print(input_param)
    # print(optimizer)
    input_param.requires_grad = True
    optimizer.requires_grad = True

    print('Optimizing...')
    epoch = [0]

    while epoch[0] < num_epochs:
        def closure():
            input_param.clamp_(0, 1)
            input_param.requires_grad = True
            model(input_param)
            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score = style_score + sl.backward()
            for cl in content_loss_list:
                content_score = content_score + cl.backward()

            epoch[0] = epoch[0] + 1
            if epoch[0] % 50 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score, content_score))
                print()
            return style_score + content_score

        optimizer.step(closure())
        input_param.clamp_(0, 1)
    return input_param
