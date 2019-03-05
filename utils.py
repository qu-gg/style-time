"""
@file utils.py
@author qu-gg

Utility functions for use in the model
"""
import imageio
import numpy as np
import os


def get_style(name):
    image = imageio.imread("images/styles/" + name)
    image = np.reshape(image, [3, 224, 224])
    return image


def get_styles():
    style_list = []
    for filename in os.listdir("images/styles/"):
        style_list.append(filename)
    return style_list