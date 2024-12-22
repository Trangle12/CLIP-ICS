import torch
import torch.nn as nn


def cosine_dist(x, y):
    '''
    :param x: torch.tensor, 2d
    :param y: torch.tensor, 2d
    :return:
    '''

    bs1 = x.size()[0]
    bs2 = y.size()[0]

    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down

    return cosine


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



class RankingLoss:

    def __init__(self):
        pass

    def _label2similarity(self, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity, more_similar):

        if more_similar is 'smaller':
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,descending=True)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n

        elif more_similar is 'larger':
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n


class TripletLoss(RankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin, metric, indicator=False, batch_average=True):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance
        '''
        self.margin = margin
        self.metric = metric
        self.indicator = indicator
        if batch_average:
            self.reduction = 'mean'
        else:
            self.reduction = 'none'
        self.margin_loss = nn.MarginRankingLoss(margin=margin, reduction=self.reduction)

    def __call__(self, emb1, emb2, emb3, label1, label2, label3):
        '''
        :param emb1: torch.Tensor, [m, dim]
        :param emb2: torch.Tensor, [n, dim]
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [b]
        :return:
        '''

        if self.metric == 'cosine':
            mat_dist = cosine_dist(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            mat_dist = cosine_dist(emb1, emb3)
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            margin_label = -torch.ones_like(hard_p)

        elif self.metric == 'euclidean':
            mat_dist = euclidean_dist(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            mat_dist = euclidean_dist(emb1, emb3)
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            margin_label = torch.ones_like(hard_p)

        return self.margin_loss(hard_n, hard_p, margin_label)


