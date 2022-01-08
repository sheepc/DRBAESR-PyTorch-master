import logging
import math

import torch
import torch.nn as nn
from torchvision.transforms import transforms


def make_model(args, parent=False):
    return DAE(args)


# -------------------------------------UpSampler-----------------------------------------#
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


# -------------------------------------LR------------------------------------------------#
# 编码器的头部为多层卷积层
class LREncoderHead(nn.Module):
    def __init__(self, args):
        super(LREncoderHead, self).__init__()
        self.scale = args.scale
        # conv_dims
        self.conv_dims = [3, 16, 32]

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
                    nn.BatchNorm2d(conv_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = conv_dim
        self.encoderConv = nn.Sequential(*modules)

    def forward(self, x):
        out = self.encoderConv(x)
        logging.info("LREncoderHead output shape is : %s" % str(out.shape))
        return out


# 编码器的尾部为多层感知机
class LREncoderTail(nn.Module):
    def __init__(self, args):
        super(LREncoderTail, self).__init__()
        self.scale = args.scale
        # fc_dims
        self.conv_dims = [256, 128, 64, 32]

        modules = []
        in_channels = self.conv_dims[0]
        for conv_dim in self.conv_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=conv_dim,
                              kernel_size=1),
                    nn.Conv2d(in_channels=conv_dim,
                              out_channels=conv_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(conv_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = conv_dim

        self.encoderMlp = nn.Sequential(*modules)

    def forward(self, x):
        out = self.encoderMlp(x)
        logging.info("LREncoderTail output shape is: %s" % str(out.shape))
        return out


# 解码器头部为多层感知机，与编码器的尾部多层感知机结构对称
class LRDecoderHead(nn.Module):
    def __init__(self, args):
        super(LRDecoderHead, self).__init__()
        self.scale = args.scale
        conv_dims = [256, 128, 64, 32]
        conv_dims.reverse()
        modules = []
        # Build Decoder_head
        in_channel = conv_dims[0]
        for conv_dim in conv_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channel,
                              out_channels=conv_dim,
                              kernel_size=1),
                    nn.Conv2d(in_channels=conv_dim,
                              out_channels=conv_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(conv_dim),
                    nn.LeakyReLU()
                )
            )
            in_channel = conv_dim

        self.decoderMlp = nn.Sequential(*modules)

    def forward(self, z):
        out = self.decoderMlp(z)
        logging.info("LRDecoderHead output shape is %s" % str(out.shape))
        return out

class LRDecoderTail(nn.Module):
    def __init__(self, args):
        super(LRDecoderTail, self).__init__()
        self.scale = args.scale
        conv_dims = [256, 128, 64, 32, 16, 8]

        modules = []
        for i in range(len(conv_dims) - 1):
            modules.append(
                nn.Sequential(
                    default_conv(conv_dims[i], 4 * conv_dims[i], 3, bias=True),
                    nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=conv_dims[i], out_channels=conv_dims[i+1], kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(in_channels=conv_dims[i+1],
                              out_channels=conv_dims[i+1],
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(conv_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        modules.append(
            nn.Sequential(
                default_conv(conv_dims[-1], 3, 3, bias=True),
                nn.Tanh()
            )
        )
        # modules.append(
        #     nn.Sequential(
        #         nn.ConvTranspose2d(in_channels=conv_dims[-1],
        #                            out_channels=conv_dims[-1],
        #                            kernel_size=3,
        #                            stride=2,
        #                            padding=1,
        #                            output_padding=1),
        #         nn.BatchNorm2d(conv_dims[-1]),
        #         nn.LeakyReLU(),
        #         nn.Conv2d(in_channels=conv_dims[-1],
        #                   out_channels=3,
        #                   kernel_size=2,
        #                   padding=1),
        #         nn.Tanh())
        # )
        self.decoderConv = nn.Sequential(*modules)

    def forward(self, z):
        out = self.decoderConv(z)
        logging.info("LRDecoderTail output shape is %s" % str(out.shape))
        return out


# ----------------------------------------HR----------------------------------------------#
# 编码器的头部为多层卷积层
class HREncoderHead(nn.Module):
    def __init__(self, args):
        super(HREncoderHead, self).__init__()
        self.scale = args.scale
        # conv_dims
        self.conv_dims = [16, 32, 64, 128, 256]

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
                    nn.BatchNorm2d(conv_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = conv_dim
        scale = args.scale[0]
        while (scale // 2) != 0:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.conv_dims[-1],
                              out_channels=self.conv_dims[-1],
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.BatchNorm2d(self.conv_dims[-1]),
                    nn.LeakyReLU()
                )
            )
            scale = scale // 2

        self.encoderConv = nn.Sequential(*modules)

    def forward(self, x):
        out = self.encoderConv(x)
        logging.info("HREncoderHead output shape is : %s" % str(out.shape))
        return out


# 编码器的尾部为多层感知机
class HREncoderTail(nn.Module):
    def __init__(self, args):
        super(HREncoderTail, self).__init__()
        self.scale = args.scale
        # fc_dims
        self.conv_dims = [256, 128, 64, 32]

        modules = []
        in_channels = self.conv_dims[0]
        for conv_dim in self.conv_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=conv_dim,
                              kernel_size=1),
                    nn.Conv2d(in_channels=conv_dim,
                              out_channels=conv_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(conv_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = conv_dim

        self.encoderMlp = nn.Sequential(*modules)

    def forward(self, x):
        out = self.encoderMlp(x)
        out = out * 255
        logging.info("HREncoderTail output shape is : %s" % str(out.shape))
        return out


# 解码器头部为多层感知机，与编码器的尾部多层感知机结构对称
class HRDecoderHead(nn.Module):
    def __init__(self, args):
        super(HRDecoderHead, self).__init__()
        self.scale = args.scale
        conv_dims = [32, 64, 128, 256]
        modules = []
        # Build Decoder_head
        in_channel = conv_dims[0]
        for conv_dim in conv_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channel,
                              out_channels=conv_dim,
                              kernel_size=1),
                    nn.Conv2d(in_channels=conv_dim,
                              out_channels=conv_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(conv_dim),
                    nn.LeakyReLU()
                )
            )
            in_channel = conv_dim

        self.decoderMlp = nn.Sequential(*modules)

    def forward(self, z):
        out = self.decoderMlp(z)
        logging.info("HRDecoderHead output shape is : %s" % str(out.shape))
        return out


# 解码器尾部为多层反卷积层
class HRDecoderTail(nn.Module):
    def __init__(self, args):
        super(HRDecoderTail, self).__init__()
        self.scale = args.scale
        conv_dims = [256, 128, 64, 32, 16, 8]

        modules = []
        scale = args.scale[0]

        for i in range(len(conv_dims) - 1):
            modules.append(
                nn.Sequential(
                    default_conv(conv_dims[i], 4 * conv_dims[i], 3, bias=True),
                    nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=conv_dims[i], out_channels=conv_dims[i+1],
                              kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(in_channels=conv_dims[i+1],
                              out_channels=conv_dims[i+1],
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(conv_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        while (scale // 2) != 0:
            modules.append(
                nn.Sequential(
                    default_conv(conv_dims[-1], 4 * conv_dims[-1], 3, bias=True),
                    nn.PixelShuffle(2),
                    # nn.BatchNorm2d(conv_dims[-1]),
                    nn.LeakyReLU()
                )
            )
            scale = scale // 2

        modules.append(
            nn.Sequential(
                default_conv(conv_dims[-1], 3, 3, bias=True),
                nn.Tanh()
            )
        )
        self.decoderConv = nn.Sequential(*modules)

    def forward(self, z):
        out = self.decoderConv(z)
        out = out * 255
        logging.info("HRDecoderTail output shape is : " + str(out.shape))
        return out


# Double Auto Encoder
class DAE(nn.Module):
    def __init__(self, args):
        super(DAE, self).__init__()
        args.conv_dims = [32, 64, 128, 256, 512]
        args.fc_dims = [256, 128, 64]

        # lr auto encoder
        self.lr_encoderHead = LREncoderHead(args)
        self.lr_encoderTail = LREncoderTail(args)
        self.lr_decoderHead = LRDecoderHead(args)
        self.lr_decoderTail = LRDecoderTail(args)

        # hr auto encoder
        self.hr_encoderHead = HREncoderHead(args)
        self.hr_encoderTail = HREncoderTail(args)
        self.hr_decoderHead = HRDecoderHead(args)
        self.hr_decoderTail = HRDecoderTail(args)

    def forward(self, lr, hr) -> list:
        # lr
        # lr encoder
        logging.info("---------------------------lr-------------------------------")
        logging.info("lr.shape is " + str(lr.shape))
        lr_encoder_conv = self.lr_encoderHead(lr)
        lr_z = self.lr_encoderTail(lr_encoder_conv)

        logging.info("lr_z shape is %s" % str(lr_z.shape))
        # print("lr_z is ", lr_z)
        # lr decoder
        lr_decoder_mlp = self.lr_decoderHead(lr_z)
        logging.info("lr_decoder_mlp shape is %s" % str(lr_decoder_mlp.shape))
        lr_recons = self.lr_decoderTail(lr_decoder_mlp)

        # hr
        # hr encoder
        logging.info("---------------------------hr-------------------------------")
        logging.info("hr.shape is %s" % str(hr.shape))
        hr_encoder_conv = self.hr_encoderHead(hr)
        hr_z = self.hr_encoderTail(hr_encoder_conv)
        logging.info("hr_z shape is %s" % str(hr_z.shape))
        # print("hr_z is ", hr_z)
        # hr decoder
        hr_decoder_mlp = self.hr_decoderHead(hr_z)
        logging.info("hr_decoder_mlp shape is %s" % str(hr_decoder_mlp.shape))
        hr_recons = self.hr_decoderTail(hr_decoder_mlp)

        # sr
        logging.info("---------------------------sr-------------------------------")
        sr_conv = self.hr_decoderHead(lr_z)
        sr = self.hr_decoderTail(sr_conv)
        logging.info("sr.shape is %s" % str(sr.shape))

        return [lr_recons, hr_recons, sr, lr_z, hr_z]