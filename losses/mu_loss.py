import math
import torch
import torch.nn as nn


def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = torch.tanh(hdr_image / norm_value)
    return mu_tonemap(bounded_hdr, mu)


def mu_tonemap(hdr_image, mu=5000):
    return torch.log(1 + mu * hdr_image) / torch.log(1 + torch.tensor(mu))

class MU(nn.Module):
    _requires_cuda = False
    _record = {"L1":0}
    def __init__(self, opt):
        super(MU, self).__init__()

        self.opt= opt
        self.gamma = 2.24
        self.percentile = 99
        self.loss = nn.L1Loss()

    def forward(self, pred, label):
        prediction = pred.pow(self.gamma)
        label = label.pow(self.gamma)

        norm_perc = torch.quantile(pred.type(torch.float64), self.percentile / 100).item()
        mu_label = tanh_norm_mu_tonemap(label, norm_perc)
        mu_prediction = tanh_norm_mu_tonemap(prediction, norm_perc)
        result = self.loss(mu_prediction, mu_label)
        self._record["L1"] = result.item()
        return result

