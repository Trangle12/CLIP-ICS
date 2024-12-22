import torch

def make_optimizer_1stage(model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = 0.00035
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
    optimizer = getattr(torch.optim, "Adam")(params)
    return optimizer



# def make_optimizer_2stage(model):
#     params = []
#     keys = []
#     for key, value in model.named_parameters():
#         if "text_encoder" in key:
#             value.requires_grad_(False)
#             continue
#         if "prompt_learner" in key:
#             value.requires_grad_(False)
#             continue
#         if not value.requires_grad:
#             continue
#         lr = 0.00035
#         weight_decay = 0.0005
#         if "bias" in key:
#             lr = 0.00035 * 1.0
#             weight_decay = 0.0005
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#         keys += [key]
#     optimizer = getattr(torch.optim, "Adam")(params)
#
#     return optimizer

def make_vit_optimizer_stage2(model):
    """
    Create ViT optimizer.

    Params:
        cfg: Config instance.
        model: The model to be optimized.
    Returns:
        An optimizer.
    """

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.00035
        weight_decay = 0.0005
        if "bias" in key:
            lr = 0.00035 * 1.0
            weight_decay = 0.0005
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, 'Adam')(params)

    return optimizer

def make_optimizer_2stage( model_net):
    params = []
    keys = []
    for key, value in model_net.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = 0.00035
        weight_decay = 0.0005
        if "bias" in key:
            lr = 0.00035 * 2
            weight_decay = 0.0005

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    optimizer_net = getattr(torch.optim, 'Adam')(params)
    return optimizer_net
