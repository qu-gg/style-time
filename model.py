"""
@file model.py
@author qu-gg

VGG-based network for use in real-time neural style transfer
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
import imageio as im
import numpy as np
import utils


def conv_layer(in_channels, out_channels, kernel):
    """
    Represents a single conv layer within the network, consisting of a 2D convolution, activating it with
    ReLU, and then applying a 2D maxpool
    :param in_channels: input number of filters for the layer
    :param out_channels: output number of filters for the layer
    :param kernel: size of each kernel
    :return: nn.Sequential of the layer
    """
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    return layer


class VGG(nn.Module):
    """
    Model of the network, a VGG-based network that consists of 6 conv layers \
    and a softmax activation at the end
    """
    def __init__(self):
        super(VGG, self).__init__()

        self.conv_1 = conv_layer(3, 64, 2)
        self.conv_2 = conv_layer(64, 128, 2)
        self.conv_3 = conv_layer(128, 256, 2)
        self.conv_4 = conv_layer(256, 512, 2)
        self.conv_5 = conv_layer(512, 512, 2)
        self.conv_6 = nn.Conv2d(512, 1, kernel_size=2, padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = torch.softmax(self.conv_6(x), dim=3)
        return x


image = utils.get_style(utils.DOCTOR)
image = torch.Tensor([image])
net = VGG()
output = net(image)
print(output)