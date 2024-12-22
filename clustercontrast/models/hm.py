import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
import collections
from torch.cuda import amp
from .losses import CrossEntropyLabelSmooth


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * \
                ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM_Avg(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * \
                ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_avg(inputs, indexes, features, momentum=0.5):
    return CM_Avg.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hybrid(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(
                    ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * \
                ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index+nums] = ctx.features[index+nums] * \
                ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

        return grad_inputs, None, None, None


def cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_WgtMean(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum, tau_w):
        ctx.features = features
        ctx.momentum = momentum
        ctx.tau_w = tau_w
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(
                    ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance)

            # distances = F.normalize(torch.stack(distances, dim=0), dim=0)
            distances = torch.stack(distances, dim=0)
            w = F.softmax(- distances / ctx.tau_w, dim=0)
            features = torch.stack(features, dim=0)
            w_mean = w.unsqueeze(1).expand_as(features) * features
            w_mean = w_mean.sum(dim=0)
            # mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * \
                ctx.momentum + (1 - ctx.momentum) * w_mean
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None, None


def cm_wgtmean(inputs, indexes, features, momentum=0.5, tau_w=0.09):
    return CM_WgtMean.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), torch.Tensor([tau_w]).to(inputs.device))

class CM(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Easy(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmax(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_easy(inputs, indexes, features, momentum=0.5):
    return (CM_Easy.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)))


class CM_Hybrid_v2(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum, num_instances):
        ctx.features = features
        ctx.momentum = momentum
        ctx.num_instances = num_instances
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features) // (ctx.num_instances + 1)
        # print(len(ctx.features))
        # print(nums)
        # print(targets)

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        updated = set()
        for k, (instance_feature, index) in enumerate(zip(inputs, targets.tolist())):
            batch_centers[index].append(instance_feature)
            if index not in updated:
                indexes = [index + nums * i for i in range(1, (targets == index).sum() + 1)]
                ctx.features[indexes] = inputs[targets == index]
                updated.add(index)

        for index, features in batch_centers.items():
            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()

        # for index, features in batch_centers.items():
        #     distances = []
        #     for feature in features:
        #         distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
        #         distances.append(distance.cpu().numpy())
        #
        #     median = np.argmax(np.array(distances))
        #     ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
        #     ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None, None


def cm_hybrid_v2(inputs, indexes, features, momentum=0.5, num_instances=16, *args):
    return CM_Hybrid_v2.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), num_instances)

