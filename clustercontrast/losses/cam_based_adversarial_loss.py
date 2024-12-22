import torch
import torch.nn.functional as F
from torch import nn
import math
import pdb
# from losses.gather import GatherLayer


class CamBasedAdversarialLoss(nn.Module):
    """ Clothes-based Adversarial Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """
    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon

    def forward(self, inputs, targets, positive_mask,pos_cam_mask):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with 
                the same identity as the anchor sample are defined as positive clothes classes and their mask 
                values are 1. The clothes classes with different identities from the anchor sample are defined 
                as negative clothes classes and their mask values in positive_mask are 0.
        """
  

        inputs = self.scale * inputs
        negtive_mask = 1 - positive_mask


        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = inputs - log_sum_exp_pos_and_all_neg

        mask = self.epsilon / (positive_mask.sum(1, keepdim=True)) * positive_mask
        loss = (- mask * log_prob).sum(1).mean()

        return loss




class arcAdversarialLoss(nn.Module):
    """ Clothes-based Adversarial Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
        margin (float): margin for the arcface loss. market 0.7 duke 0.3 msmt 0.5
    """
    def __init__(self, scale=16, epsilon=1 , margin = 0.7, tau = 0.2):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon
        self.margin = margin
        self.tau = tau

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, inputs, targets, positive_mask, pos_cam_mask, pos_accu):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with 
                the same identity as the anchor sample are defined as positive clothes classes and their mask 
                values are 1. The clothes classes with different identities from the anchor sample are defined 
                as negative clothes classes and their mask values in positive_mask are 0.
        """

        size = len(targets)
        # output = inputs

        #target替換為cos
        cosine = inputs.clamp(-1, 1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    
        gt = output[torch.arange(0, output.shape[0]), targets].view(-1, 1)
        
        for pos_mask_line_index,pos_mask_line in enumerate(pos_cam_mask):
            pos_cam_example = output[pos_mask_line_index][pos_mask_line.bool()]
            output[pos_mask_line_index][pos_mask_line.bool()] = (self.tau + 1.0) * pos_cam_example + self.tau
        output.scatter_(1, targets.data.view(-1, 1), gt)

        output = self.scale * output
        negtive_mask = 1 - positive_mask
        identity_mask = torch.zeros(output.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        exp_logits = torch.exp(output)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim = True) + exp_logits)
        log_prob = output - log_sum_exp_pos_and_all_neg


        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim = True) * positive_mask
        loss = (- mask * log_prob).sum(1).mean()
        return loss

      

      
        




class MutiCamBasedLoss(nn.Module):
    """ Clothes-based Adversarial Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """
    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with 
                the same identity as the anchor sample are defined as positive clothes classes and their mask 
                values are 1. The clothes classes with different identities from the anchor sample are defined 
                as negative clothes classes and their mask values in positive_mask are 0.
        """
        inputs = self.scale * inputs
        negtive_mask = 1 - positive_mask
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = inputs - log_sum_exp_pos_and_all_neg


        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_prob).sum(1).mean()

        return loss