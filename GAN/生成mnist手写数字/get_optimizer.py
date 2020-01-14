# coding=gbk
from torch import optim
from generator import Generator


def get_optimizer(MODEL, learning_rate=0.001, betas=(0.5, 0.999)):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    Optimizer = optim.Adam(MODEL.parameters(), lr=learning_rate, betas=betas)
    return Optimizer


# model = Generator(10, 780)
# optimizer = get_optimizer(MODEL=model)
# print(optimizer)
