import math
import torch
import torch.nn as nn


def mu_tonemap(hdr_image, mu=5000):
    return torch.log(1 + mu * hdr_image) / math.log(1 + mu)


class Base(nn.Module):
    _requires_cuda = False
    _record = {"L1":0}
    def __init__(self, opt):
        super(Base, self).__init__()

        self.opt= opt
        self.loss = nn.L1Loss()

    def forward(self, pred, label):
        mu_prediction = mu_tonemap(pred)
        mu_label = mu_tonemap(label)
        
        result = self.loss(mu_prediction, mu_label)
        self._record["L1"] = result.item()
        return result

