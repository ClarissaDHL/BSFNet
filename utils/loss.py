import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.utils import *
from torch.autograd import Variable
import numpy as np

class contrastive_loss(nn.Module):
    """
    no-change，0
    change，1
    """
    def __init__(self, margin1=0.1, margin2=2.0, eps=1e-6):
        super(contrastive_loss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps

    def forward(self, x1, x2, label):
        diff = torch.abs(x1 - x2)
        dist_sq = torch.pow(diff + self.eps, 2).sum(dim=1)
        dist = torch.sqrt(dist_sq)

        total = np.prod(label.size())
        refer = 1-label
        neg_dis = torch.clamp(dist - self.margin1, min=0.0)
        loss1 = refer * (neg_dis.pow(2))
        loss_1 = torch.sum(loss1)

        pos_dis = torch.clamp(self.margin2 - dist, min=0.0)
        loss2 = label * (pos_dis.pow(2)) * 10.0
        loss_2 = torch.sum(loss2)
        loss = (loss_1 +  loss_2) / total
        return loss

class BCL_v2(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，0
    change，1
    """
    def __init__(self, margin1=0.1, margin2=2.0, eps=1e-6):
        super(BCL_v2, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps

    def forward(self, x1, x2, label):
        diff = torch.abs(x1 - x2)
        dist_sq = torch.pow(diff + self.eps, 2).sum(dim=1)
        dist = torch.sqrt(dist_sq)

        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==0).float())+0.0001

        refer = 1-label
        neg_dis = torch.clamp(dist - self.margin1, min=0.0)
        loss1 = refer * (neg_dis.pow(2))
        loss_1 = torch.sum(loss1) /neg_num

        pos_dis = torch.clamp(self.margin2 - dist, min=0.0)
        loss2 = label * (pos_dis.pow(2))
        loss_2 = torch.sum(loss2) / pos_num
        loss = loss_1 +  loss_2
        return loss

class BCLwithUncertainty_v1(nn.Module):
    def __init__(self, margin1=0.1, margin2=2.0, eps=1e-6, gamma=2):
        super(BCLwithUncertainty_v1, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps
        self.gamma = gamma
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, label):
        diff = x1 - x2
        dist = torch.pow(diff + self.eps, 2).sum(dim=1)
        dist_sq = torch.sqrt(dist)
        pos_num = torch.sum((label==1).float()) + 0.001
        neg_num = torch.sum((label==0).float()) + 0.001

        smooth_label = torch.pow(self.avgpool(label), self.gamma)
        smooth_refer = torch.pow((1-self.avgpool(label)), self.gamma)
        refer = 1 - label

        neg_dis = torch.clamp(dist_sq - self.margin1, min=0.0)
        loss_neg = (refer + smooth_refer) * neg_dis
        loss_1 = torch.sum(loss_neg) / neg_num

        pos_dis = torch.clamp(self.margin2 - dist_sq, min=0.0)
        loss_pos = (label + smooth_label) * pos_dis
        loss_2 = torch.sum(loss_pos) / pos_num
        loss_dis = loss_1 + loss_2
        return loss_dis

class BCLwithUncertainty_v2(nn.Module):
    def __init__(self, margin1=0.1, margin2=1.8, eps=1e-6):
        super(BCLwithUncertainty_v2, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, label):
        diff = x1 - x2
        dist = torch.pow(diff + self.eps, 2).sum(dim=1)
        dist_sq = torch.sqrt(dist)
        pos_num = torch.sum((label==1).float()) + 0.001
        neg_num = torch.sum((label==0).float()) + 0.001

        x = self.avgpool(label)
        weight = 0.8 - 4 * torch.pow(x, 2) + 4 * x

        refer = 1 - label
        neg_dis = torch.clamp(dist_sq - self.margin1, min=0.0)
        loss_neg = refer * neg_dis * weight
        loss_1 = torch.sum(loss_neg) / neg_num

        pos_dis = torch.clamp(self.margin2 - dist_sq, min=0.0)
        loss_pos = label * pos_dis * weight
        loss_2 = torch.sum(loss_pos) / pos_num
        loss_dis = loss_1 + loss_2
        return loss_dis

def cross_entropy_2d(predict, target):
    """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
    """
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0)
    assert predict.size(2) == target.size(1)
    assert predict.size(3) == target.size(2)
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1,2).transpose(2,3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, size_average=True)
    return loss

class IOUloss_v1(nn.Module):
    def __init__(self, margin1=0.1, margin2=2.0, eps=1e-6):
        super(IOUloss_v1, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps

    def forward(self, x1, x2, label):
        diff = x1 - x2
        dist = torch.pow(diff + self.eps, 2).sum(dim=1)
        dist_sq = torch.sqrt(dist)

        predict = dist_sq[:,:,:]>1.0
        gt = label[:,:,:]==1.0

        insection = (predict & gt).float()
        union = (predict | gt).float()
        iou_loss = 1 - (torch.sum(insection) / (torch.sum(union)))

        return iou_loss

class IOUloss_v2(nn.Module):
    def __init__(self, eps=1e-6):
        super(IOUloss_v2, self).__init__()
        self.eps = eps

    def forward(self, output, label):
        _, predicted = torch.max(output.data, dim=1)

        predict = predicted[:,:,:]==1
        gt = label[:,:,:]==1.0

        insection = (predict & gt).float()
        union = (predict | gt).float()
        iou_loss = 1 - (torch.sum(insection) / (torch.sum(union)))
        return iou_loss
