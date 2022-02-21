from metrics.distance import *
import torch
from torch.nn import functional as F
from losses.triplet_hard_loss import TripletLoss
from losses.triplet_hard_loss import TripletHardLoss
from metrics.distance import *
import torch.nn as nn

def aligned_loss(global_feat, labels, local_feat, logits, normalize_feature=True, margin=0.5):
    """
    Args:
        global_feat: pytorch Variable, shape [N, C]
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the 
        Channel dimension
    Returns:
        loss: pytorch Variable, with shape [1]
        ===================
        For Mutual Learning
        ===================
        dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    g_tri_loss = TripletLoss(margin=margin)
    l_tri_loss = TripletLoss(margin=margin)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    if normalize_feature:
        global_feat = F.normalize(global_feat, p=2, dim=1)
        local_feat = F.normalize(local_feat, p=2, dim=1)
    # shape [N, N]
    # global_dist_mat = compute_distance_matrix(global_feat, global_feat, 'euclidean')
    # gloss = g_tri_loss(global_dist_mat, labels)
    # local_dist_mat = local_dist(local_feat, local_feat)
    # lloss = l_tri_loss(local_dist_mat, labels)
    # gloss, lloss = triplet_loss(global_feat, labels, local_feat)
    gloss, p_inds, n_inds, g_dist_ap, g_dist_an, g_dist_mat = global_loss(g_tri_loss, global_feat, labels)
    lloss, l_dist_ap, l_dist_an = local_loss(l_tri_loss, local_feat, p_inds, n_inds, labels)
    class_loss = cross_entropy_loss(logits, labels)
    return gloss + lloss + class_loss

# for alignedReID

"""Numpy version of euclidean distance, shortest distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input      
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


def shortest_dist(dist_mat):
    """Parallel version.
  Args:
    dist_mat: numpy array, available shape
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`
      1) scalar
      2) numpy array, with shape [N]
      3) numpy array with shape [*]
  """
    m, n = dist_mat.shape[:2]
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]
    # I ran into memory disaster when returning this reference! I still don't
    # know why.
    # dist = dist[-1, -1]
    dist = dist[-1, -1].copy()
    return dist

def unaligned_dist(dist_mat):
    """Parallel version.
    Args:
      dist_mat: numpy array, available shape
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
      dist: three cases corresponding to `dist_mat`
        1) scalar
        2) numpy array, with shape [N]
        3) numpy array with shape [*]
    """

    m = dist_mat.shape[0]
    dist = np.zeros_like(dist_mat[0])
    for i in range(m):
        dist[i] = dist_mat[i][i]
    dist = np.sum(dist, axis=0).copy()
    return dist


def meta_local_dist(x, y, aligned):
    """
  Args:
    x: numpy array, with shape [m, d]
    y: numpy array, with shape [n, d]
  Returns:
    dist: scalar
  """
    eu_dist = compute_dist(x, y, 'euclidean')
    dist_mat = (np.exp(eu_dist) - 1.) / (np.exp(eu_dist) + 1.)
    if aligned:
        dist = shortest_dist(dist_mat[np.newaxis])[0]
    else:
        dist = unaligned_dist(dist_mat[np.newaxis])[0]
    return dist


def parallel_local_dist(x, y, aligned):
    """Parallel version.
  Args:
    x: numpy array, with shape [M, m, d]
    y: numpy array, with shape [N, n, d]
  Returns:
    dist: numpy array, with shape [M, N]
  """
    M, m, d = x.shape
    N, n, d = y.shape
    x = x.reshape([M * m, d])
    y = y.reshape([N * n, d])
    # shape [M * m, N * n]
    dist_mat = compute_dist(x, y, type='euclidean')
    dist_mat = (np.exp(dist_mat) - 1.) / (np.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.reshape([M, m, N, n]).transpose([1, 3, 0, 2])
    # shape [M, N]
    if aligned:
        dist_mat = shortest_dist(dist_mat)
    else:
        dist_mat = unaligned_dist(dist_mat)
    return dist_mat


def local_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [M, m, d]
        y: pytorch Variable, with shape [N, n, d]
    Returns:
        dist: pytorch Variable, with shape [M, N]
    """
    M, m, d = x.size()
    N, n, d = y.size()
    x = x.contiguous().view(M * m, d)
    y = y.contiguous().view(N * n, d)
    # shape [M * m, N * n]
    dist_mat = euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
    # shape [M, N]
    dist_mat = shortest_dist(dist_mat)
    return dist_mat

def batch_local_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [N, m, d]
        y: pytorch Variable, with shape [N, n, d]
    Returns:
        dist: pytorch Variable, with shape [N]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    # shape [N, m, n]
    dist_mat = batch_euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [N]
    dist = shortest_dist(dist_mat.permute(1, 2, 0))
    return dist


def shortest_dist(dist_mat):
    """Parallel version.
    Args:
        dist_mat: pytorch Variable, available shape:
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
        dist: three cases corresponding to `dist_mat`:
        1) scalar
        2) pytorch Variable, with shape [N]
        3) pytorch Variable, with shape [*]
    """
    m, n = dist_mat.size()[:2]
    # Just offering some reference for accessing intermediate distance.
    dist = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:
                dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
    dist = dist[-1][-1]
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
        return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N]; 
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples, 
    thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos][:N].view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg][:N].view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze( 0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
        ind[is_pos][:N].view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
        ind[is_neg][:N].view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def batch_euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [Batch size, Local part, Feature channel]
        y: pytorch Variable, with shape [Batch size, Local part, Feature channel]
    Returns:
        dist: pytorch Variable, with shape [Batch size, Local part, Local part]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    N, m, d = x.size()
    N, n, d = y.size()

    # shape [N, m, n]
    xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
    yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
    dist = xx + yy
    dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def global_loss(tri_loss, global_feat, labels, normalize_feature=True):
    """
    Args:
        tri_loss: a `TripletLoss` object
        global_feat: pytorch Variable, shape [N, C]
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the 
        Channel dimension
    Returns:
        loss: pytorch Variable, with shape [1]
        p_inds: pytorch LongTensor, with shape [N]; 
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        =============
        For Debugging
        =============
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        ===================
        For Mutual Learning
        ===================
        dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels, return_inds=True)
    loss = tri_loss(dist_ap, dist_an)
    return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat


def local_loss(
    tri_loss,
    local_feat,
    p_inds=None,
    n_inds=None,
    labels=None,
    normalize_feature=True):
    """
    Args:
        tri_loss: a `TripletLoss` object
        local_feat: pytorch Variable, shape [N, H, c] (NOTE THE SHAPE!)
        p_inds: pytorch LongTensor, with shape [N]; 
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the 
        Channel dimension
    
    If hard samples are specified by `p_inds` and `n_inds`, then `labels` is not 
    used. Otherwise, local distance finds its own hard samples independent of 
    global distance.
    
    Returns:
        loss: pytorch Variable,with shape [1]
        =============
        For Debugging
        =============
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        ===================
        For Mutual Learning
        ===================
        dist_mat: pytorch Variable, pairwise local distance; shape [N, N]
    """
    if normalize_feature:
        local_feat = normalize(local_feat, axis=-1)
    if p_inds is None or n_inds is None:
        dist_mat = local_dist(local_feat, local_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels, return_inds=False)
        loss = tri_loss(dist_ap, dist_an)
        return loss, dist_ap, dist_an, dist_mat
    else:
        dist_ap = batch_local_dist(local_feat, local_feat[p_inds])
        dist_an = batch_local_dist(local_feat, local_feat[n_inds])
        loss = tri_loss(dist_ap, dist_an)
        return loss, dist_ap, dist_an