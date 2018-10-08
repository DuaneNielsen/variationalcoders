from unittest import TestCase
from mentalitystorm.transforms import *
import torchvision.transforms as TVT
import imageio
from pathlib import Path
from mentalitystorm.data import GymSimulatorDataset
from mentalitystorm.policies import VCPolicy, RandomPolicy
from mentalitystorm.config import config
from mentalitystorm.storage import Storeable
import gym

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
        invader = ColorMask(lower=[120, 125, 25], upper=[140, 140, 130], append=True)
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


    def test_invaders_on_sim(self):
        env = gym.make('SpaceInvaders-v4')

        policy = RandomPolicy(env)

        dataset = GymSimulatorDataset(env, policy, 3000, output_in_numpy_format=True)

        screen, observation, action, reward, done, _ = dataset.__getitem__(0)

        imageio.imwrite('test_image.png', screen)

        image = imageio.imread('test_image.png')
        #image = imageio.imread(Path(r'C:\data\SpaceInvaders-v4\images\raw_v1\all\pic0001_0015.png'))

        #shots = ColorMask(lower=[128, 128, 128], upper=[255, 255, 255], append=True)
        #player = ColorMask(lower=[30, 100, 40], upper=[70, 180, 70], append=True)
        #cut_player = SetRange(0, 60, 0, 210, [4])
        invader = ColorMask(lower=[120, 125, 25], upper=[140, 140, 130], append=True)
        #cut_invader = SetRange(0, 30, 0, 210, [5])
        #barrier = ColorMask(lower=[120, 74, 30], upper=[190, 100, 70], append=True)
        select = SelectChannels([3, 4, 5, 6])
        select = SelectChannels([3])

        #segmentor = TVT.Compose([shots, player, cut_player, invader, cut_invader, barrier, select])
        #segmentor = TVT.Compose([shots, player, invader, barrier, select])
        segmentor = TVT.Compose([invader, select])
        segments = segmentor(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', image)
        cv2.imshow('shots', segments[:, :, 0])
        # cv2.imshow('player', segments[:, :, 1])
        # cv2.imshow('invader', segments[:, :, 2])
        # cv2.imshow('barrier', segments[:, :, 3])
        k = cv2.waitKey()

    def test_colors(self):
        env = gym.make('SpaceInvaders-v4')

        policy = RandomPolicy(env)

        dataset = GymSimulatorDataset(env, policy, 3000, output_in_numpy_format=True)

        screen, observation, action, reward, done, _ = dataset.__getitem__(0)

        imageio.imwrite('test_image.png', screen)

        image = imageio.imread('test_image.png')

        print((image - screen).mean())
        print((image - observation).mean())
        print((screen - observation).mean())

        cv2.imshow('frame', image-screen)
        k = cv2.waitKey()
