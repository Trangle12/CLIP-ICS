import torch
import torch.nn.functional as F
from torch import nn
from torch import distributed as dist
import math
from clustercontrast.losses.softmax_loss import CrossEntropyLabelSmooth

class CoscamLoss(nn.Module):
    """ CosFace Loss based on the predictions of classifier.

    Reference:
        Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """
    def __init__(self, scale=16, margin=0.1, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets, mask, pos_cam_mask):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """

    
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        cam_inputs = pos_cam_mask.cuda() * inputs.cuda()+(1 - pos_cam_mask).cuda() * -10000

        gt = inputs[torch.arange(0, inputs.shape[0]), targets].view(-1, 1)

        mask = cam_inputs >= gt
        #if len(selected_idxs) != 0 :
        #    mask[selected_idxs] = ((mask[selected_idxs] + pos_mask.cuda())>=1)

        

        hard_example = inputs[mask]
        inputs[mask] = (0.012 + 1.0) * hard_example + 0.012

        '''
            84.38之前每加
        '''
        final_gt = gt
        inputs.scatter_(1, targets.data.view(-1, 1), final_gt)

        output = self.s * (inputs - one_hot * self.m)

        return F.cross_entropy(output, targets)



# class ArcFaceLoss(nn.Module):
#     """ CosFace Loss based on the predictions of classifier.

#     Reference:
#         Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

#     Args:
#         scale (float): scaling factor.
#         margin (float): pre-defined margin.
#     """
#     def __init__(self, scale=16, margin=0.1, **kwargs):
#         super().__init__()
#         self.s = scale
#         self.m = margin

#         self.cos_m = math.cos(margin)
#         self.sin_m = math.sin(margin)

#         # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
#         self.th = math.cos(math.pi - margin)
#         self.mm = math.sin(math.pi - margin) * margin

#     def forward(self, inputs, targets, pos_cam_mask):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (batch_size)
#         """

#         cosine = inputs.clamp(-1, 1)

#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m

#         phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, targets.view(-1, 1), 1)
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
#         output = output * self.s

#         return F.cross_entropy(output, targets) 


# class ArcFaceLoss(nn.Module):
#     """ CosFace Loss based on the predictions of classifier.

#     Reference:
#         Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

#     Args:
#         scale (float): scaling factor.
#         margin (float): pre-defined margin.
#     """
#     def __init__(self, scale=16, margin=0.1, **kwargs):
#         super().__init__()
#         self.s = scale
#         self.m = margin

#         self.cos_m = math.cos(margin)
#         self.sin_m = math.sin(margin)

#         # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
#         self.th = math.cos(math.pi - margin)
#         self.mm = math.sin(math.pi - margin) * margin

#     def forward(self, inputs, targets, pos_cam_mask):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (batch_size)
#         """

#         one_hot = torch.zeros_like(inputs)
#         one_hot.scatter_(1, targets.view(-1, 1), 1.0)

#         output = self.s * (inputs - one_hot * self.m)

#         return F.cross_entropy(output, targets)



class ArcFaceLoss(nn.Module):
    """ CosFace Loss based on the predictions of classifier.

    Reference:
        Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """
    def __init__(self, scale=16, margin=0.1, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin
        self.ce_ls = CrossEntropyLabelSmooth(num_classes=3262)
        # self.epsilon = epsilon




    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """

        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        output = self.s * (inputs - one_hot * self.m)




        return F.cross_entropy(output, targets)
        #return  self.ce_ls(output, targets)

       
       

class CosFaceLoss(nn.Module):
    """ CosFace Loss based on the predictions of classifier.

    Reference:
        Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """
    def __init__(self, scale=16, margin=0.1, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        output = self.s * (inputs - one_hot * self.m)

        return F.cross_entropy(output, targets)

# class PairwiseCosFaceLoss(nn.Module):
#     """ CosFace Loss among sample pairs.

#     Reference:
#         Sun et al. Circle Loss: A Unified Perspective of Pair Similarity Optimization. In CVPR, 2020.

#     Args:
#         scale (float): scaling factor.
#         margin (float): pre-defined margin.
#     """
#     def __init__(self, scale=16, margin=0):
#         super().__init__()
#         self.s = scale
#         self.m = margin

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: sample features (before classifier) with shape (batch_size, feat_dim)
#             targets: ground truth labels with shape (batch_size)
#         """
#         # l2-normalize
#         inputs = F.normalize(inputs, p=2, dim=1)

#         # gather all samples from different GPUs as gallery to compute pairwise loss.
#         gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
#         gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
#         m, n = targets.size(0), gallery_targets.size(0)

#         # compute cosine similarity
#         similarities = torch.matmul(inputs, gallery_inputs.t())
        
#         # get mask for pos/neg pairs
#         targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
#         mask = torch.eq(targets, gallery_targets.T).float().cuda()
#         mask_self = torch.zeros_like(mask)
#         rank = dist.get_rank()
#         mask_self[:, rank * m:(rank + 1) * m] += torch.eye(m).float().cuda()
#         mask_pos = mask - mask_self
#         mask_neg = 1 - mask

#         scores = (similarities + self.m) * mask_neg - similarities * mask_pos
#         scores = scores * self.s
        
#         neg_scores_LSE = torch.logsumexp(scores * mask_neg - 99999999 * (1 - mask_neg), dim=1)
#         pos_scores_LSE = torch.logsumexp(scores * mask_pos - 99999999 * (1 - mask_pos), dim=1)

#         loss = F.softplus(neg_scores_LSE + pos_scores_LSE).mean()

#         return loss