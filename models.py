from mentalitystorm import Storeable, BaseVAE
import torch
import torch.nn as nn
import torch.nn.functional as F





class ConvVAE4Fixed(Storeable, BaseVAE):
    """
    input_shape is a tuple of (height, width)
    """
    def __init__(self, input_shape, z_size, variational=True, input_channels=3, output_channels=3, first_kernel=5,
                 first_stride=2, second_kernel=5, second_stride=2):
        self.input_shape = input_shape
        self.z_size = z_size
        encoder = self.Encoder(input_shape, z_size, input_channels, first_kernel, first_stride, second_kernel, second_stride)
        decoder = self.Decoder(z_size, encoder.z_shape, output_channels, first_kernel, first_stride, second_kernel, second_stride)
        BaseVAE.__init__(self, encoder, decoder, variational)
        Storeable.__init__(self)

    class Encoder(nn.Module):
        def __init__(self, input_shape, z_size, channels=3, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)
            # batchnorm in autoencoding is a thing
            # https://arxiv.org/pdf/1602.02282.pdf

            from mentalitystorm.util import conv_output_shape

            # encoder
            self.e_conv1 = nn.Conv2d(channels, 32, kernel_size=first_kernel, stride=first_stride)
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
        def __init__(self, z_size, z_shape, channels=3, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)

            # decoder
            self.d_conv1 = nn.ConvTranspose2d(z_size, 128, kernel_size=z_shape, stride=1)
            self.d_bn1 = nn.BatchNorm2d(128)

            self.d_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=second_kernel, stride=second_stride,
                                              output_padding=(1, 0))
            self.d_bn2 = nn.BatchNorm2d(128)

            self.d_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=second_kernel, stride=second_stride,
                                              output_padding=(0, 1))  # output_padding=1)
            self.d_bn3 = nn.BatchNorm2d(32)

            self.d_conv4 = nn.ConvTranspose2d(32, channels, kernel_size=first_kernel, stride=first_stride, output_padding=1)

        def forward(self, z):
            decoded = F.relu(self.d_bn1(self.d_conv1(z)))
            decoded = F.relu(self.d_bn2(self.d_conv2(decoded)))
            decoded = F.relu(self.d_bn3(self.d_conv3(decoded)))
            decoded = self.d_conv4(decoded)
            return torch.sigmoid(decoded)

class ConvVAE4FixedSub(BaseVAE):
    """
    input_shape is a tuple of (height,width)
    """
    def __init__(self, input_shape, z_size, variational=True, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
        self.input_shape = input_shape
        self.z_size = z_size
        encoder = self.Encoder(input_shape, z_size, 3, first_kernel, first_stride, second_kernel, second_stride)
        decoder = self.Decoder(z_size, encoder.z_shape, 3, first_kernel, first_stride, second_kernel, second_stride)
        BaseVAE.__init__(self, encoder, decoder, variational)


class MultiChannelConv(Storeable):
    def __init__(self, input_shape, z_size, variational=True, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
        self.vae = ConvVAE4Fixed(self, input_shape, z_size, variational, first_kernel, first_stride, second_kernel, second_stride)
        self.shot_vae = ConvVAE4Fixed(self, input_shape, z_size, variational, 1, first_kernel, first_stride, second_kernel, second_stride)
        Storeable.__init__(self)

    def forward(self, image):
        vae_channels = image[:, 0:2]
        vae_out = self.vae(vae_channels)
        shot_channel = image[:, 3]
        shot_out = self.shot_vae(shot_channel)
        output = torch.cat((vae_out, shot_out), dim=1)
        return output


class AtariVAE2DLatent(Storeable, BaseVAE):
    """
    input_shape is a tuple of (height, width)
    """
    def __init__(self, input_shape, z_size, variational=True, input_channels=3, output_channels=3, first_kernel=5,
                 first_stride=2, second_kernel=5, second_stride=2):
        self.input_shape = input_shape
        self.z_size = z_size
        encoder = self.Encoder(input_shape, z_size, input_channels, first_kernel, first_stride, second_kernel, second_stride)
        decoder = self.Decoder(z_size, encoder.z_shape, output_channels, first_kernel, first_stride, second_kernel, second_stride)
        BaseVAE.__init__(self, encoder, decoder, variational)
        Storeable.__init__(self)

    class Encoder(nn.Module):
        def __init__(self, input_shape, z_size, channels=3, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)
            # batchnorm in autoencoding is a thing
            # https://arxiv.org/pdf/1602.02282.pdf

            from mentalitystorm.util import conv_output_shape


            s1 = 32
            s2 = 64
            # encoder
            self.e_conv1 = nn.Conv2d(channels, s1, kernel_size=first_kernel, stride=first_stride)
            self.e_bn1 = nn.BatchNorm2d(s1)
            output_shape = conv_output_shape(input_shape, kernel_size=first_kernel, stride=first_stride)

            self.e_conv2 = nn.Conv2d(s1, s2, kernel_size=second_kernel, stride=second_stride)
            self.e_bn2 = nn.BatchNorm2d(s2)
            output_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

            self.e_conv3 = nn.Conv2d(s2, s2, kernel_size=second_kernel, stride=second_stride)
            self.e_bn3 = nn.BatchNorm2d(s2)
            output_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

            self.e_conv4 = nn.Conv2d(s2, s2, kernel_size=2, stride=2)
            self.e_bn4 = nn.BatchNorm2d(s2)
            output_shape = conv_output_shape(output_shape, kernel_size=2, stride=2)

            self.e_mean = nn.Conv2d(s2, z_size, kernel_size=2, stride=2)
            self.e_logvar = nn.Conv2d(s2, z_size, kernel_size=2, stride=2)
            self.z_shape = conv_output_shape(output_shape, kernel_size=2, stride=2)

        def forward(self, x):
            encoded = F.relu(self.e_bn1(self.e_conv1(x)))
            encoded = F.relu(self.e_bn2(self.e_conv2(encoded)))
            encoded = F.relu(self.e_bn3(self.e_conv3(encoded)))
            encoded = F.relu(self.e_bn4(self.e_conv4(encoded)))
            mean = (torch.sigmoid(self.e_mean(encoded)) * 6) - 3
            logvar = torch.sigmoid(self.e_logvar(encoded)) * 3
            return mean, logvar

    class Decoder(nn.Module):
        def __init__(self, z_size, z_shape, channels=3, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)

            s1 = 32
            s2 = 64

            self.d_convz = nn.ConvTranspose2d(z_size, s2, kernel_size=2, stride=2, output_padding=(1, 0))
            self.d_bnz = nn.BatchNorm2d(s2)

            self.d_conv1 = nn.ConvTranspose2d(s2, s2, kernel_size=2, stride=2, output_padding=1)
            self.d_bn1 = nn.BatchNorm2d(s2)
            # decoder
            #self.d_conv1 = nn.ConvTranspose2d(z_size, 128, kernel_size=z_shape, stride=1)
            #self.d_bn1 = nn.BatchNorm2d(128)

            self.d_conv2 = nn.ConvTranspose2d(s2, s2, kernel_size=second_kernel, stride=second_stride,
                                              output_padding=(1, 0))
            self.d_bn2 = nn.BatchNorm2d(s2)

            self.d_conv3 = nn.ConvTranspose2d(s2, s1, kernel_size=second_kernel, stride=second_stride,
                                              output_padding=(0, 1))  # output_padding=1)
            self.d_bn3 = nn.BatchNorm2d(s1)

            self.d_conv4 = nn.ConvTranspose2d(s1, channels, kernel_size=first_kernel, stride=first_stride, output_padding=1)

        def forward(self, z):
            decoded = F.relu(self.d_bnz(self.d_convz(z)))
            decoded = F.relu(self.d_bn1(self.d_conv1(decoded)))
            decoded = F.relu(self.d_bn2(self.d_conv2(decoded)))
            decoded = F.relu(self.d_bn3(self.d_conv3(decoded)))
            decoded = self.d_conv4(decoded)
            return torch.sigmoid(decoded)
