from __future__ import division, absolute_import
import torch
import torch.nn as nn
from .triplet_hard_loss import TripletHardLoss

class MGNLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Args:
        margin (float, optional): margin for triplet. Default is 0.5.
    """

    def __init__(self, margin=1.2):
        super(MGNLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """

        cross_entropy = nn.CrossEntropyLoss()
        triplet = TripletHardLoss(self.margin)

        tri_loss = [triplet(input_f, targets) for input_f in inputs[0:3]]
        tri_loss = sum(tri_loss)/len(tri_loss)

        cross_loss = [cross_entropy(input_f, targets) for input_f in inputs[3:]]
        cross_loss = sum(cross_loss)/len(cross_loss)

        loss = tri_loss + 2 * cross_loss

        return loss
