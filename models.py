from mentalitystorm import Storeable, BaseVAE
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
input_shape is a tuple of (height,width)
"""
class ConvVAE4Fixed(Storeable, BaseVAE):
    def __init__(self, input_shape, z_size, variational=True, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
        self.input_shape = input_shape
        self.z_size = z_size
        encoder = self.Encoder(input_shape, z_size, first_kernel, first_stride, second_kernel, second_stride)
        decoder = self.Decoder(z_size, encoder.z_shape, first_kernel, first_stride, second_kernel, second_stride)
        BaseVAE.__init__(self, encoder, decoder, variational)
        Storeable.__init__(self)


    class Encoder(nn.Module):
        def __init__(self, input_shape, z_size, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)
            # batchnorm in autoencoding is a thing
            # https://arxiv.org/pdf/1602.02282.pdf

            from mentalitystorm.util import conv_output_shape

            # encoder
            self.e_conv1 = nn.Conv2d(3, 32, kernel_size=first_kernel, stride=first_stride)
            self.e_bn1 = nn.BatchNorm2d(32)
            output_shape = conv_output_shape(input_shape, kernel_size=first_kernel, stride=first_stride)

            self.e_conv2 = nn.Conv2d(32, 128, kernel_size=second_kernel, stride=second_stride)
            self.e_bn2 = nn.BatchNorm2d(128)
            output_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

            self.e_conv3 = nn.Conv2d(128, 128, kernel_size=second_kernel, stride=second_stride)
            self.e_bn3 = nn.BatchNorm2d(128)
            self.z_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

            self.e_mean = nn.Conv2d(128, z_size, kernel_size=self.z_shape, stride=1)
            self.e_logvar = nn.Conv2d(128, z_size, kernel_size=self.z_shape, stride=1)

        def forward(self, x):
            encoded = F.relu(self.e_bn1(self.e_conv1(x)))
            encoded = F.relu(self.e_bn2(self.e_conv2(encoded)))
            encoded = F.relu(self.e_bn3(self.e_conv3(encoded)))
            mean = self.e_mean(encoded)
            logvar = self.e_logvar(encoded)
            return mean, logvar

    class Decoder(nn.Module):
        def __init__(self, z_size, z_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)

            # decoder
            self.d_conv1 = nn.ConvTranspose2d(z_size, 128, kernel_size=z_shape, stride=1)
            self.d_bn1 = nn.BatchNorm2d(128)

            self.d_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=second_kernel, stride=second_stride)# , output_padding=(1,0))
            self.d_bn2 = nn.BatchNorm2d(128)

            self.d_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=second_kernel, stride=second_stride, output_padding=1) #, output_padding=(0,1))
            self.d_bn3 = nn.BatchNorm2d(32)

            self.d_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=first_kernel, stride=first_stride, output_padding=1)

        def forward(self, z):
            decoded = F.relu(self.d_bn1(self.d_conv1(z)))
            decoded = F.relu(self.d_bn2(self.d_conv2(decoded)))
            decoded = F.relu(self.d_bn3(self.d_conv3(decoded)))
            decoded = self.d_conv4(decoded)
            return torch.sigmoid(decoded)