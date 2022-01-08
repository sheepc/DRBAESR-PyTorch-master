import torch.nn as nn


class JSDivLoss(nn.Module):
    def __init__(self, args):
        super(JSDivLoss, self).__init__()

    def forward(self, input, target):
        js = 0.5 * nn.KLDivLoss(input, (input + target) / 2) + 0.5 * nn.KLDivLoss(target, (input + target) / 2)
        return js
