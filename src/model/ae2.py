import logging
import math

import torch
import torch.nn as nn
from torchvision.transforms import transforms


def make_model(args, parent=False):
    return AE2(args)


# -------------------------------------UpSampler-----------------------------------------#
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


# -------------------------------------LR------------------------------------------------#
# 编码器的头部为多层卷积层
class LREncoder(nn.Module):
    def __init__(self, args):
        super(LREncoder, self).__init__()
        self.scale = args.scale
        # conv_dims
        self.conv_dims = [32, 64, 128]

        modules = []
        in_channels = 3
        # Build Encoder_conv
        for conv_dim in self.conv_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=conv_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.Conv2d(in_channels=conv_dim,
                              out_channels=conv_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    # nn.BatchNorm2d(conv_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = conv_dim
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        out = self.encoder(x)
        return out


class LRDecoder(nn.Module):
    def __init__(self, args):
        super(LRDecoder, self).__init__()
        self.scale = args.scale
        conv_dims = [128, 64, 32, 3]
        modules = []
        for i in range(len(conv_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=conv_dims[i], out_channels=conv_dims[i+1], kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(in_channels=conv_dims[i+1], out_channels=conv_dims[i+1], kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_channels=conv_dims[i+1], out_channels=conv_dims[i+1] * 4, kernel_size=1, stride=1, padding=0),
                    nn.PixelShuffle(2),  # c -> c/4; h->hx2; w->wx2
                    # nn.BatchNorm2d(conv_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        i = len(conv_dims) - 2
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels=conv_dims[i], out_channels=conv_dims[i + 1], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=conv_dims[i + 1], out_channels=conv_dims[i + 1], kernel_size=3, stride=1,
                          padding=1),
                nn.Conv2d(in_channels=conv_dims[i + 1], out_channels=conv_dims[i + 1] * 4, kernel_size=1, stride=1,
                          padding=0),
                nn.PixelShuffle(2),  # c -> c/4; h->hx2; w->wx2
                # nn.BatchNorm2d(conv_dims[i + 1]),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        out = self.decoder(z)
        return out


# ----------------------------------------HR----------------------------------------------#
# 编码器的头部为多层卷积层
class HREncoder(nn.Module):
    def __init__(self, args):
        super(HREncoder, self).__init__()
        self.scale = args.scale
        # conv_dims
        self.conv_dims = [32, 64, 128, 128]

        modules = []
        in_channels = 3
        # Build Encoder_conv
        for conv_dim in self.conv_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=conv_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.Conv2d(in_channels=conv_dim,
                              out_channels=conv_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    # nn.BatchNorm2d(conv_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = conv_dim
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        out = self.encoder(x)
        # logging.info("HREncoder output shape is : %s" % str(out.shape))
        return out


# 解码器头部为多层感知机，与编码器的尾部多层感知机结构对称
class HRDecoder(nn.Module):
    def __init__(self, args):
        super(HRDecoder, self).__init__()
        self.scale = args.scale
        conv_dims = [128, 128, 64, 32, 3]

        modules = []
        for i in range(len(conv_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=conv_dims[i], out_channels=conv_dims[i+1], kernel_size=1, stride=1,
                              padding=0),
                    nn.Conv2d(in_channels=conv_dims[i+1], out_channels=conv_dims[i+1], kernel_size=3, stride=1,
                              padding=1),
                    nn.Conv2d(in_channels=conv_dims[i+1], out_channels=conv_dims[i+1] * 4, kernel_size=1, stride=1,
                              padding=0),
                    nn.PixelShuffle(2),
                    # nn.BatchNorm2d(conv_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        i = len(conv_dims) - 2
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels=conv_dims[i], out_channels=conv_dims[i + 1], kernel_size=1, stride=1,
                          padding=0),
                nn.Conv2d(in_channels=conv_dims[i + 1], out_channels=conv_dims[i + 1], kernel_size=3, stride=1,
                          padding=1),
                nn.Conv2d(in_channels=conv_dims[i + 1], out_channels=conv_dims[i + 1] * 4, kernel_size=1, stride=1,
                          padding=0),
                nn.PixelShuffle(2),
                # nn.BatchNorm2d(conv_dims[i + 1]),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        out = self.decoder(z)
        # logging.info("HRDecoder output shape is %s" % str(out.shape))
        return out


# Double Auto Encoder
class AE2(nn.Module):
    def __init__(self, args):
        super(AE2, self).__init__()
        args.conv_dims = [32, 64, 128, 256, 512]
        args.fc_dims = [256, 128, 64]

        # lr auto encoder
        self.lr_encoder = LREncoder(args)
        self.lr_decoder = LRDecoder(args)

        # hr auto encoder
        self.hr_encoder = HREncoder(args)
        self.hr_decoder = HRDecoder(args)

    def forward(self, lr, hr) -> list:
        # lr
        # lr encoder
        logging.info("---------------------------lr-------------------------------")
        logging.info("AE2 lr.shape is " + str(lr.shape))
        # print("AE2 lr {}", lr)
        lr_z = self.lr_encoder(lr)
        # print("AE2 lr_encoder parameters")
        # for name, param in self.lr_encoder.named_parameters():
        #     print(name, param.shape)

        logging.info("AE2 lr_z shape is %s" % str(lr_z.shape))
        # print("AE2 lr_z {}", lr_z)
        # lr decoder
        lr_recons = self.lr_decoder(lr_z)
        logging.info("AE2 lr_recons shape is %s" % str(lr_recons.shape))
        # print("lr_recons is ", lr_recons)
        # hr
        # hr encoder
        logging.info("---------------------------hr-------------------------------")
        logging.info("AE2 hr.shape is %s" % str(hr.shape))
        hr_z = self.hr_encoder(hr)
        logging.info("AE2 hr_z shape is %s" % str(hr_z.shape))
        # hr decoder
        hr_recons = self.hr_decoder(hr_z)
        logging.info("AE2 hr_recons shape is %s" % str(hr_recons.shape))

        # sr
        # logging.info("---------------------------sr-------------------------------")
        # sr_l = lr
        # sr_z = self.lr_encoder(sr_l)
        # sr = self.hr_decoder(sr_z)
        # logging.info("sr.shape is %s" % str(sr.shape))

        return [lr_recons, hr_recons]
