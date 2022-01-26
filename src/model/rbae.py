import logging
import math

import torch
import torch.nn as nn
from torchvision.transforms import transforms


def make_model(args, parent=False):
    return RBAE(args)


## Residual Block (RB)
# n_feat -> n_feat
# h -> h
# w -> w
class RB(nn.Module):
    def __init__(self, n_feat, factor):
        super(RB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat * factor, kernel_size=3, stride=1, padding=1))
        modules_body.append(nn.ReLU())
        modules_body.append(nn.Conv2d(n_feat * factor, n_feat, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##Residual Block Group(RBG)
# n_feat -> n_feat
# h -> h
# w -> w
class RBG(nn.Module):
    def __init__(self, n_rb, n_feat, factor):
        super(RBG, self).__init__()
        modules_body = []
        for i in range(n_rb):
            modules_body.append(RB(n_feat, factor))
        self.body = nn.Sequential(*modules_body)
        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = self.body(x)
        res += x
        res = self.conv(res)
        return res


# Down Sample Block
# n_feat -> n_feat
# h->h/sqrt(factor)
# w->w/sqrt(factor)
class DownSampler(nn.Module):
    def __init__(self, n_feat, factor):
        super(DownSampler, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat * factor, kernel_size=3, stride=2, padding=1))
        modules_body.append(nn.Conv2d(n_feat * factor, n_feat * factor, kernel_size=3, stride=1, padding=1))
        modules_body.append(nn.LeakyReLU())
        modules_body.append(nn.Conv2d(n_feat * factor, n_feat, kernel_size=1, stride=1, padding=0))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


# Up Sample Block
# # n_feat -> n_feat
# # h->h/sqrt(factor)
# # w->w/sqrt(factor)
class UpSampler(nn.Module):
    def __init__(self, n_feat, factor):
        super(UpSampler, self).__init__()
        modules_body = []
        modules_body.append(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat * factor, kernel_size=3, stride=1, padding=1))
        modules_body.append(
            nn.Conv2d(in_channels=n_feat * factor, out_channels=n_feat * factor, kernel_size=1, stride=1, padding=0))
        modules_body.append(nn.PixelShuffle(int(math.sqrt(factor))))
        # modules_body.append(nn.Sigmoid())

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


# -------------------------------------LR------------------------------------------------#
# 编码器的头部为多层卷积层
class LREncoder(nn.Module):
    def __init__(self, args):
        super(LREncoder, self).__init__()
        self.args = args
        self.n_rb = 6
        self.n_rbg = 3
        self.n_feat = 64
        self.factor = 1

        modules_body = []
        self.conv_1 = nn.Conv2d(3, self.n_feat, kernel_size=3, stride=1, padding=1)

        for i in range(self.n_rbg):
            modules_body.append(RBG(self.n_rb, self.n_feat, self.factor))
        self.body = nn.Sequential(*modules_body)

        self.conv_2 = nn.Conv2d(self.n_feat, self.n_feat, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        res = self.body(x)
        res += x
        res = self.conv_2(res)
        return res


class LRDecoder(nn.Module):
    def __init__(self, args):
        super(LRDecoder, self).__init__()
        self.args = args
        self.n_rb = 6
        self.n_rbg = 3
        self.n_feat = 64
        self.factor = 1

        modules_body = []
        self.conv_1 = nn.Conv2d(self.n_feat, self.n_feat, kernel_size=3, stride=1, padding=1)

        for i in range(self.n_rbg):
            modules_body.append(RBG(self.n_rb, self.n_feat, self.factor))
        self.body = nn.Sequential(*modules_body)

        self.conv_2 = nn.Conv2d(self.n_feat, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        res = self.body(x)
        res += x
        res = self.conv_2(res)
        # self.activation = nn.Sigmoid()
        if self.args.normalized:
            res = torch.clamp(res, 0, 1)
        else:
            res = torch.clamp(res, 0, 255)
        return res


# ----------------------------------------HR----------------------------------------------#
# 编码器的头部为多层卷积层
class HREncoder(nn.Module):
    def __init__(self, args):
        super(HREncoder, self).__init__()
        self.args = args
        self.n_rb = 6
        self.n_rbg = 3
        self.n_feat = 64
        self.factor = 1
        self.scale = 2

        self.conv_1 = nn.Conv2d(3, self.n_feat, kernel_size=3, stride=1, padding=1)
        modules_body = []
        for i in range(self.n_rbg//2):
            modules_body.append(RBG(self.n_rb, self.n_feat, self.factor))

        modules_body.append(DownSampler(self.n_feat, self.factor))
        for i in range(self.n_rbg - self.n_rbg//2):
            modules_body.append(RBG(self.n_rb, self.n_feat, self.factor))

        self.body = nn.Sequential(*modules_body)
        self.conv_2 = nn.Conv2d(self.n_feat, self.n_feat, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.body(x)
        x = self.conv_2(x)
        return x


class HRDecoder(nn.Module):
    def __init__(self, args):
        super(HRDecoder, self).__init__()
        self.args = args
        self.n_rb = 6
        self.n_rbg = 3
        self.n_feat = 64
        self.factor = 1
        self.scale = 2

        self.conv_1 = nn.Conv2d(self.n_feat, self.n_feat, kernel_size=3, stride=1, padding=1)
        modules_body = []
        for i in range(self.n_rbg // 2):
            modules_body.append(RBG(self.n_rb, self.n_feat, self.factor))

        modules_body.append(UpSampler(self.n_feat, self.scale*self.scale))

        for i in range(self.n_rbg - self.n_rbg // 2):
            modules_body.append(RBG(self.n_rb, self.n_feat, self.factor))

        self.body = nn.Sequential(*modules_body)
        self.conv_2 = nn.Conv2d(self.n_feat, 3, kernel_size=3, stride=1, padding=1)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.body(x)
        x = self.conv_2(x)
        if self.args.normalized:
            x = torch.clamp(x, 0, 1)
        else:
            x = torch.clamp(x, 0, 255)
        return x


# Residual Block Double AutoEncoder
class RBAE(nn.Module):
    def __init__(self, args):
        super(RBAE, self).__init__()
        self.args = args

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
        # logging.info("AE2 hr_z shape is %s" % str(hr_z.shape))
        # hr decoder
        hr_recons = self.hr_decoder(hr_z)
        logging.info("AE2 hr_recons shape is %s" % str(hr_recons.shape))

        # sr
        # logging.info("---------------------------sr-------------------------------")
        sr_l = lr.clone()
        sr_z = self.lr_encoder(sr_l)
        sr = self.hr_decoder(sr_z)
        logging.info("sr.shape is %s" % str(sr.shape))

        return [lr_recons, hr_recons, lr_z, hr_z, sr]
