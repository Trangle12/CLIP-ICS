import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .pooling import build_pooling_layer

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class build_rn50(nn.Module):
    def __init__(self, args, camera_num):
        super(build_rn50, self).__init__()
        self.model_name = 'RN50'
        self.in_planes = 2048
        self.in_planes_proj = 1024
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        input_size_train = args.image_size
        stride_size = [16, 16]
        self.h_resolution = int((input_size_train[0] - 16) // stride_size[0] + 1)
        self.w_resolution = int((input_size_train[1] - 16) // stride_size[1] + 1)
        self.vision_stride_size = stride_size[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.gap = build_pooling_layer('gem')

        if args.dataset == 'market1501':
            num_classes = 3262
        elif args.dataset == 'msmt17':
            num_classes = 4821
        else:
            num_classes = 2196

        self.num_classes = num_classes
        self.prompt_learner = PromptLearner(args.dataset, num_classes,clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x=None, label=None, get_image=False, get_text=False ):
        if get_text == True:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features
        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            return image_features_proj[0]

        image_features_last, image_features, image_features_proj = self.image_encoder(x)

        img_feature = self.gap(image_features).view(x.shape[0], -1)
        img_feature_proj = image_features_proj[0]

        feat = self.bottleneck(img_feature)
        #feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            return F.normalize(feat, dim=1), img_feature_proj  #F.normalize(feat, dim=1)
        else:
            return F.normalize(feat, dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if not self.training and 'classifier' in i:
                continue # ignore classifier weights in evaluation
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



def make_model(args,camera_num):
    model = build_rn50(args,camera_num)
    return model

import clip.clip as clip

def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class PromptLearner(nn.Module):
    def __init__(self,dataset_name, num_class, dtype, token_embedding):
        super().__init__()

        ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        if dataset_name == 'market1501' or dataset_name == 'msmt17':
            n_ctx = 5   # market msmt17 5  duke 6
        elif dataset_name == 'dukemtmc':
            n_ctx = 6

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

