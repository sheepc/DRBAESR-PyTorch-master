import torch.nn as nn
from scipy.stats import wasserstein_distance


class WSDivLoss(nn.Module):
    def __init__(self, args):
        super(WSDivLoss, self).__init__()

    def forward(self, input, target):
        return wasserstein_distance(input, target)
