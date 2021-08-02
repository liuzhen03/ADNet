import torch
import torch.nn as nn
import math
import numpy as np


def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = torch.tanh(hdr_image / norm_value)
    return mu_tonemap(bounded_hdr, mu)


def mu_tonemap(hdr_image, mu=5000):
    return torch.log(1 + mu * hdr_image) / math.log(1 + mu)


class mu_loss(object):
    def __init__(self, gamma=2.24, percentile=99):
        self.gamma = gamma
        self.percentile = percentile

    def __call__(self, pred, label):
        hdr_linear_ref = pred ** self.gamma
        hdr_linear_res = label ** self.gamma
        norm_perc = np.percentile(hdr_linear_ref.data.cpu().numpy().astype(np.float32), self.percentile)
        mu_pred = tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc)
        mu_label = tanh_norm_mu_tonemap(hdr_linear_res, norm_perc)
        return nn.L1Loss()(mu_pred, mu_label)

