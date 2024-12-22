import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from .losses import CrossEntropyLabelSmooth
from torch.cuda import amp
from clustercontrast.losses.focal_loss import FocalLoss
from .hm import cm, cm_hard, cm_avg, cm_hybrid_v2, tccl


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.1, mode=' CM',num_instances=32):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode
        self.num_instances = num_instances
        self.cross_entropy = nn.CrossEntropyLoss().cuda()

        self.cross_entropy_ls = CrossEntropyLabelSmooth(num_classes=num_samples).cuda()

    def forward(self, inputs , targets,  cameras , cam=False):
            inputs = F.normalize(inputs, dim=1).cuda()

            if cam:
                outputs = cm_hybrid_v2(inputs, targets, self.features, self.momentum, self.num_instances)
                out_list = torch.chunk(outputs, self.num_instances + 1, dim=1)
                out = torch.stack(out_list[1:], dim=0)
                neg = torch.max(out, dim=0)[0]
                pos = torch.min(out, dim=0)[0]
                mask = torch.zeros_like(out_list[0]).scatter_(1, targets.unsqueeze(1), 1)
                logits = mask * pos + (1 - mask) * neg
                loss_centroid = self.cross_entropy(out_list[0] / self.temp, targets)
                loss_instance = self.cross_entropy(logits / self.temp, targets)
                return loss_centroid , loss_instance
            else:
                outputs = cm_hard(inputs, targets, self.features, self.momentum)
                outputs /= self.temp
                loss = self.cross_entropy(outputs, targets)

                return loss
