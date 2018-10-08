from unittest import TestCase
from mentalitystorm.transforms import *
import torchvision.transforms as TVT
import imageio
from pathlib import Path

class TestTransforms(TestCase):
    def test_colormask(self):

        image = imageio.imread(Path(r'C:\data\SpaceInvaders-v4\images\raw_v1\all\pic0001_0015.png'))

        mask = ColorMask([30, 100, 40], [70, 180, 70])(image)
        mask = SetRange(0, 60, 0, 210, [0])(mask)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', image)
        cv2.imshow('mask', mask)
        k = cv2.waitKey()

    def test_segment(self):
        shots = ColorMask(lower=[128, 128, 128], upper=[255, 255, 255], append=True)
        player = ColorMask(lower=[30, 100, 40], upper=[70, 180, 70], append=True)
        cut = SetRange(0, 60, 0, 210, [4])
        select = SelectChannels([3, 4])

        segmentor = TVT.Compose([shots, player, cut, select])

        image = imageio.imread(Path(r'C:\data\SpaceInvaders-v4\images\raw_v1\all\pic0001_0015.png'))
        segments = segmentor(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', image)
        cv2.imshow('shots', segments[:, :, 0])
        cv2.imshow('player', segments[:, :, 1])
        k = cv2.waitKey()

    def test_invaders(self):
        shots = ColorMask(lower=[128, 128, 128], upper=[255, 255, 255], append=True)
        player = ColorMask(lower=[30, 100, 40], upper=[70, 180, 70], append=True)
        cut_player = SetRange(0, 60, 0, 210, [4])
        invader = ColorMask(lower=[120, 125, 30], upper=[140, 140, 130], append=True)
        cut_invader = SetRange(0, 30, 0, 210, [5])
        barrier = ColorMask(lower=[120, 74, 30], upper=[190, 100, 70], append=True)
        select = SelectChannels([3, 4, 5, 6])

        segmentor = TVT.Compose([shots, player, cut_player, invader, cut_invader, barrier, select])

        image = imageio.imread(Path(r'C:\data\SpaceInvaders-v4\images\raw_v1\all\pic0001_0015.png'))
        segments = segmentor(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', image)
        cv2.imshow('shots', segments[:, :, 0])
        cv2.imshow('player', segments[:, :, 1])
        cv2.imshow('invader', segments[:, :, 2])
        cv2.imshow('barrier', segments[:, :, 3])
        k = cv2.waitKey()
