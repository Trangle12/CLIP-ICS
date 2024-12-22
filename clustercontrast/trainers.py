from __future__ import print_function, absolute_import
import copy
import time
import torch
from .losses.cam_based_adversarial_loss import arcAdversarialLoss
from .losses.cosface_loss import ArcFaceLoss
from .losses.triplet_loss_stb import TripletLoss
from .utils.meters import AverageMeter
from torch.cuda import amp
import numpy as np
import torch.nn.functional as F
from clustercontrast.losses.original_mcnl_loss import MCNL_Loss_global
from clustercontrast.losses.softmax_loss import CrossEntropyLabelSmooth
from clustercontrast.losses.make_loss import make_loss

class TrainerFp16(object):
    """
    trainer with FP16 forwarding.
    """
    def __init__(self, args , encoder,id_count_each_cameras, cam_classifier=None,  memory = None , cam_memorys=None,new_labels=None, cluster_text_features = None, cam2accu = None,pseudo2accu_mask = None, cam_text_features = None, nums_class = 4821,  inter_cam_memory = None, all_cam = 6):
        super(ViTTrainerFp16, self).__init__()
        self.args = args
        self.encoder = encoder
        self.memory = memory
        self.cam_classifier = cam_classifier
        self.cluster_text_features = cluster_text_features
        self.cam_text_features = cam_text_features
        self.cam_memorys = cam_memorys
        self.inter_cam_memory = inter_cam_memory
        self.cam2accu = cam2accu
        self.pseudo2accu_mask = pseudo2accu_mask
        self.new_labels = new_labels
        self.device = 'cuda'
        self.nums_class = nums_class
        self.xent_sup = CrossEntropyLabelSmooth(self.nums_class).cuda()
        print("label smooth on, numclasses:", self.nums_class)
        self.xent_intra = []
        self.id_count_each_cameras = id_count_each_cameras
        for i in self.id_count_each_cameras:
            self.xent_intra.append(CrossEntropyLabelSmooth(i).cuda())
        self.margin = 0.3
        print("using triplet loss with margin:{}".format(self.margin))
        self.criterion_triplet = TripletLoss(self.margin, 'euclidean')  # default margin=0.3
        #############################  ADV  ##########################################
        #epsilon (float): a trade-off hyper-parameter.market 0.8 duke 0.8 msmt 1.0
        #margin (float): margin for the arcface loss. market 0.7 duke 0.3 msmt 0.5
        if self.args.dataset == 'market1501':
            self.epsilon = 0.8
            self.margin_adv = 0.7
            self.loss_weight =  0.6
            self.inter_text_loss_epoch = 10
            self.start_adv_epoch = 40
            self.end_adv_epoch = 50
        elif self.args.dataset == 'dukemtmc':
            self.epsilon = 0.8
            self.margin_adv = 0.3
            self.loss_weight = 0.6  # all 0.8  market1501 0.6
            self.inter_text_loss_epoch = 40
            self.start_adv_epoch = 40
            self.end_adv_epoch = 60
        else:
            self.epsilon = 0.8   # 0.8 Best
            self.margin_adv = 0.5
            self.loss_weight = 0.0
            self.inter_text_loss_epoch = 40
            self.start_adv_epoch = 40
            self.end_adv_epoch = 60
        print("start adv epoch: {}".format(self.start_adv_epoch))
        print("using intra hard loss weight :{}".format( self.loss_weight))
        print("using inter text loss with epoch: {}".format(self.inter_text_loss_epoch))
        print("using adv loss with epsilon: {} and margin:{}".format(self.epsilon, self.margin_adv))
        self.criterion_cam = ArcFaceLoss(scale=16., margin=0)
        self.criterion_adv = arcAdversarialLoss(scale=16., epsilon=self.epsilon, margin=self.margin_adv, tau=0.2)

    def train(self, epoch, data_loader, optimizer, optimizer_cc, print_freq=10, train_iters=400, camera = 15 , intra_epoch = 5):
        self.encoder.train()
        self.cam_classifier.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_h = AverageMeter()
        cam_losses = AverageMeter()
        global_ID_losses = AverageMeter()

        end = time.time()

        # amp fp16 training
        scaler = amp.GradScaler()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next_one()
            data_time.update(time.time() - end)
            intra_loss = torch.tensor(0.).to(self.device)
            inter_loss = torch.tensor(0.).to(self.device)
            # process inputs
            imgs, labels, cams, cam_pid, g_label = self._parse_data(inputs)

            with amp.autocast(enabled=True):
                # ============== model forward ================
                x_glb, f_proj = self.encoder(imgs)

                cam_logits = self.cam_classifier(x_glb.detach())
                pos_cam_mask = self.cam2accu[cams.long()].to(self.device)
                style_loss = self.criterion_cam(cam_logits, g_label.long().to(self.device))
                optimizer_cc.zero_grad()
                scaler.scale(style_loss).backward()
                scaler.step(optimizer_cc)
                scaler.update()
                global_ID_losses.update(style_loss.item())

                if epoch < intra_epoch:
                    percam_V = []
                    for ii in range(camera):
                        num_instances = 32
                        outputs = self.cam_memorys[ii].features.detach().clone()
                        out_list = torch.chunk(outputs, num_instances + 1, dim=0)
                        percam_V.append(out_list[0])
                    # cache the per-camera memory bank before each batch training
                    for ii in range(camera):
                        target_cam = ii
                        if torch.nonzero(cams == target_cam).size(0) > 0:
                            percam_feat1 = x_glb[cams == target_cam]
                            percam_feat2 = f_proj[cams == target_cam]
                            percam_label = cam_pid[cams == target_cam]

                            loss_centroid, loss_instance  = self.cam_memorys[ii](percam_feat1.to(self.device),percam_label.long().to(self.device),cams,cam=True)
                            intra_loss += (self.loss_weight * loss_centroid + (1 - self.loss_weight) * loss_instance)

                            if self.text_loss_intra:

                                logits = percam_feat2 @ self.cam_text_features[ii].t()
                                loss_i2tc = self.xent_intra[ii](logits , percam_label.long().to(self.device))

                                intra_loss += loss_i2tc

                            cam_memory_label = torch.arange(self.id_count_each_cameras[ii]).long().to(self.device)
                            memo_trip_loss = self.criterion_triplet(percam_feat1, percam_V[ii].to(self.device),
                                                                    percam_V[ii].to(self.device), percam_label.long().to(self.device),
                                                                    cam_memory_label, cam_memory_label)
                            intra_loss += memo_trip_loss

                            TRI_LOSS = self.criterion_triplet(percam_feat1,percam_feat1, percam_feat1, percam_label.long().to(self.device),percam_label.long().to(self.device),percam_label.long().to(self.device))
                            intra_loss += TRI_LOSS

                    optimizer.zero_grad()

                    scaler.scale(intra_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    cam_losses.update(intra_loss.item())
                    torch.cuda.synchronize()
                    batch_time.update(time.time() - end)

                    end = time.time()
                    if (i + 1) % print_freq == 0:
                        print('Epoch: [{}][{}/{}]\t'
                              'Time {:.3f} ({:.3f})\t'
                              'Intra Loss {:.3f} ({:.3f})\t'
                              'Global ID Loss {:.3f} ({:.3f})\t'
                              .format(epoch, i + 1, train_iters,
                                      batch_time.val, batch_time.avg,
                                      cam_losses.val, cam_losses.avg,
                                      global_ID_losses.val,global_ID_losses.avg,
                                      ))
                else:
                    selected_idx = np.where(labels >= 0)[0]
                    if len(selected_idx) != 0:
                        f_out = x_glb[selected_idx].to(self.device)
                        labels = labels[selected_idx].to(self.device)
                        cams = cams[selected_idx].to(self.device)
                        f_proj = f_proj[selected_idx].to(self.device)
                        inter_loss += self.memory(f_out,labels,cams , cam=False)

                        if epoch >= self.inter_text_loss_epoch:
                            logits = f_proj @ self.cluster_text_features.t()
                            loss_itc = self.xent_sup(logits, labels)
                            inter_loss  += loss_itc

                        TRI_LOSS = self.criterion_triplet(f_out, f_out, f_out ,labels, labels , labels)
                        inter_loss  += TRI_LOSS

                        if epoch >= self.start_adv_epoch and epoch < self.end_adv_epoch:
                            pos_mask = self.pseudo2accu_mask[labels].cuda()
                            mask_pos_cam = pos_mask - pos_cam_mask[selected_idx].cuda()
                            the_mask = mask_pos_cam < 0
                            mask_pos_cam[the_mask] = 0
                            adv_mask = copy.deepcopy(mask_pos_cam)
                            adv_mask.scatter_(1, g_label[selected_idx].data.view(-1, 1).long(), 1)

                            all_new_pred_cam = self.cam_classifier(x_glb).to(self.device)
                            new_pred_cam = all_new_pred_cam[selected_idx]
                            selected_pos_cam_mask = self.cam2accu[cams.long()].cuda()
                            adv_loss = self.criterion_adv(new_pred_cam.to(self.device),g_label[selected_idx].long().to(self.device),
                                                         adv_mask.to(self.device), selected_pos_cam_mask, self.cam2accu)
                            inter_loss += adv_loss

                    optimizer.zero_grad()
                    scaler.scale(inter_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    losses_h.update(inter_loss.item())

                    torch.cuda.synchronize()
                    # print log
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if (i + 1) % print_freq == 0:
                        print('Epoch: [{}][{}/{}]\t'
                              'Time {:.3f} ({:.3f})\t'
                              'Inter Loss {:.3f} ({:.3f})\t'
                              'Global ID Loss {:.3f} ({:.3f})\t'
                              .format(epoch, i + 1, train_iters,
                                      batch_time.val, batch_time.avg,
                                      losses_h.val, losses_h.avg,
                                      global_ID_losses.val, global_ID_losses.avg))


    def _parse_data(self, inputs):
        ori_data = inputs
        imgs = ori_data[0]
        cams = ori_data[2]
        cam_pid = ori_data[4]
        global_label = ori_data[5] # global label
        labels = torch.tensor(self.new_labels[global_label]).long()  # predicted label

        return imgs.cuda(),labels, cams.cuda(),cam_pid.cuda(), global_label.cuda()