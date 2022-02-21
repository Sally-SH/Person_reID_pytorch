from __future__ import division, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
from metrics.distance import *


class TripletHardLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Args:
        margin (float, optional): margin for triplet. Default is 0.5.
    """

    def __init__(self, margin=0.5):
        super(TripletHardLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """

        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        return self.ranking_loss(dist_an, dist_ap, y)


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
    Loss for Person Re-Identification'."""
    def __init__(self, margin=0.5):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def __call__(self, dist_ap, dist_an):
        """
        Args:
        dist_ap: pytorch Variable, distance between anchor and positive sample, 
            shape [N]
        dist_an: pytorch Variable, distance between anchor and negative sample, 
            shape [N]
        Returns:
        loss: pytorch Variable, with shape [1]
        """
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss