"""
@file utils.py
@author qu-gg

Utility functions for use in the model
"""
import imageio
import numpy as np

DOCTOR = "doctor-who.jpg"
SPECKLED = "speckled-flowers.jpg"


def get_style(name):
    image = imageio.imread("images/styles/" + name)
    image = np.reshape(image, [3, 224, 224])
    return image
