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


class SegmentByColor(object):

    def __init__(self, lower, upper):
        self.lower = np.array(lower)
        self.upper = np.array(upper)

    def __call__(self, image):
        mask = cv2.inRange(image, lower, upper)
        image = np.append(image, mask, dim=2)
        return image


class SelectChannels(object):
    def __call__(self, image):
        return image[:, 0:3]


if __name__ == '__main__':

    lower = np.array([128, 128, 128])
    upper = np.array([255, 255, 255])

    lower = np.array([30, 100, 40])
    upper = np.array([70, 180, 70])


    # directory = Path(r'C:\data\SpaceInvaders-v4\images\raw_v1\all')
    # for file in directory.glob('pic*.png'):
    #     image = imageio.imread(file)

    image = imageio.imread(Path(r'C:\data\SpaceInvaders-v4\images\raw_v1\all\pic0001_0015.png'))
    grayscale, color, mask = color_segment(image, lower, upper)
    mask[0:60, :] = 0
    grayscale[0:60, :] = 0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    cv2.imshow('frame', image)
    cv2.imshow('mask', mask)
    cv2.imshow('res', color)
    cv2.imshow('gray', grayscale)

    k = cv2.waitKey()