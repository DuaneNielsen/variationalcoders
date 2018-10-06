import imageio
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch


class CoordConv:
    """
    Co-ord conv adds channels of x and y co-ordinates
    https://arxiv.org/abs/1807.03247
    """
    def __call__(self, image):
        height = image.shape[1]
        width = image.shape[2]
        hmap = torch.linspace(0, 1.0, height)
        hmap = hmap.repeat(1, width).reshape(width, height).permute(1, 0).unsqueeze(0)
        wmap = torch.linspace(0, 1.0, width)
        wmap = wmap.repeat(1, height).reshape(height, width).unsqueeze(0)
        image = torch.cat((image, hmap, wmap), dim=0)
        return image

def color_segment(image, lower, upper):
    mask = cv2.inRange(image, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    grayscale = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return grayscale, res, mask


class ColorMask(object):

    def __init__(self, lower, upper, append=True):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.append = append

    def __call__(self, image):
        mask = cv2.inRange(image[:, :, 0:3], self.lower, self.upper)
        mask = np.expand_dims(mask, axis=2)
        if self.append:
            mask = np.concatenate((image, mask), axis=2)
        return mask


class SelectChannels(object):
    def __init__(self, channel_list):
        self.channel_list = channel_list

    def __call__(self, image):
        return image[:, :, self.channel_list]


class SetRange(object):
    def __init__(self, x_start, x_stop, y_start, y_stop, channels, color=0):
        self.x_start = x_start
        self.x_stop = x_stop
        self.y_start = y_start
        self.y_stop = y_stop
        self.color = color
        self.channels = channels

    def __call__(self, image):
        image[self.x_start:self.x_stop, self.y_start:self.y_stop, self.channels] = self.color
        return image