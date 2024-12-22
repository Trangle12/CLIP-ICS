import torch
from torch import nn

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes=0, epsilon=0.1, topk_smoothing=False):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.k = 1 if not topk_smoothing else self.num_classes//50

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        if self.k >1:
            topk = torch.argsort(-log_probs)[:,:self.k]
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
            targets += torch.zeros_like(log_probs).scatter_(1, topk, self.epsilon / self.k)
        else:
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


# encoding: utf-8
"""

"""
# CMKIC loss
def contra_loss_topk_2feat(features, features2, labels=None, mask=None, temperature=0.07,
                           base_temperature=0.07, topk=8):
    '''
    1, positive choice topk
    2, denominator is all negative and above  choiced positive

    '''

    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))
    if len(features.shape) < 2:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 2:
        features = features.view(features.shape[0], -1)
        features2 = features2.view(features.shape[0], -1)

    # print(features.shape)

    batch_size = features.shape[0]
    if labels is None:
        raise ValueError('lables must have')
    else:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        # print(mask)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features2.T),
        temperature)
    # print(anchor_dot_contrast.shape)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    neg_mask = torch.ones_like(mask)
    #
    inf_tensor = torch.tensor(float('inf'), dtype=logits.dtype).to(device)
    masked_logits = torch.where(mask.bool(), logits, inf_tensor)
    # choice topk postive
    for i in range(mask.size(0)):
        non_zero_indices = torch.nonzero(mask[i]).squeeze()

        neg_mask[i, non_zero_indices] = 0

        if non_zero_indices.numel() > 0:
            # topk
            value, topk_indices = torch.topk(masked_logits[i], topk, largest=False)
            mask[i, :] = 0
            mask[i, topk_indices] = 1
            neg_mask[i, topk_indices] = 1

    exp_logits = torch.exp(logits)
    exp_logits = exp_logits * neg_mask  # denominator is all negative and above random choiced positive
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(batch_size, ).mean()

    return loss



