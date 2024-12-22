# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import os.path as osp
import numpy as np
import sys
import collections
import time
from datetime import timedelta
import torch
import torch.nn.functional as F
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import TrainerFp16
from clustercontrast.evaluators import test, extract_text_features
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
sys.path.append(' ')
from clustercontrast.utils.cluster_labels import img_association
def get_train_loader(loaders):
    train_loader = loaders.train_iter
    return train_loader

def get_cluster_loader(loaders):
    pe_loader = loaders.propagate_loader  # 会生成全局标签
    return pe_loader

def do_train_stage2(args,
             model,
             loaders,
             test_loader,
             optimizer,
             optimizer_cc,
             scheduler,
             id_count_each_cam,
             cameras,
            cam_classifier
             ):
    global  best_mAP
    best_mAP = 0
    start_time = time.monotonic()
    text_features = extract_text_features(model, loaders.propagate_loader)
    #########################################################
    # 获取相机的mask
    accu2cam = []
    cam2accu =[]
    cam2accu_unique = []
    before = 0
    after = 0
    for cam_i in range(loaders.cam_num):
        if cam_i < loaders.cam_num:
            after += id_count_each_cam[cam_i]
        accu2cam.append(np.arange(before, after))

        if cam_i < loaders.cam_num:
            before += id_count_each_cam[cam_i]
        cam_mask = torch.zeros(sum(id_count_each_cam))

        cam2accu.append(cam_mask.scatter_(0,torch.LongTensor(accu2cam[cam_i]),1))

        true_cam_mask = torch.zeros(loaders.cam_num)
        cam2accu_unique.append(true_cam_mask.scatter_(0,torch.LongTensor([cam_i]),1))

    cam2accu_unique = torch.stack(cam2accu_unique, dim=0)
    cam2accu = torch.stack(cam2accu, dim=0)

    #########################################################
    trainer = ViTTrainerFp16(args , model , id_count_each_cam)
    trainer.cam_classifier = cam_classifier
    for epoch in range(args.epochs):

        print('==> Create inter camera labels ')

        # select & cluster images as training set of this epochs
        propagate_loader = get_cluster_loader(loaders)
        pseudo_labels, features, global_labels, all_cams, cam_labels, new_IDs = img_association(args, model,
                                                                                           propagate_loader,
                                                                                           id_count_each_cam)  # new_label
        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        ######################################################################################
        pseudo2accu = collections.defaultdict()
        pseudo2accu_list = []
        pid2cam = []
        pid2cam_unique = []
        correct_num = 0
        for i in np.unique(pseudo_labels):
            pseudo2accu[i] = np.where(pseudo_labels == i)  # batch中每个图片的accu_label
            if i != -1:
                pseudo2accu_tensor = torch.zeros(sum(id_count_each_cam))
                pseudo2accu_tensor.scatter_(0, torch.LongTensor(np.where(pseudo_labels == i)[0]), 1)
                # 检测聚类后的标签是否存在噪声
                if len(np.unique(new_IDs[torch.where(pseudo2accu_tensor == 1)[0]])) == 1:
                    correct_num += 1
                pseudo2accu_list.append(pseudo2accu_tensor)
        pseudo2accu_list = torch.stack(pseudo2accu_list, dim=0)
        print("correct/all:", correct_num, '/', len(np.unique(pseudo_labels)))
        ########################################################################################
        @torch.no_grad()
        def generate_center_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if int(label) == -1:
                    continue
                centers[int(labels[i])].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        @torch.no_grad()
        def generate_instance_features(pseudo_labels, features, num_cluster, num_instances):
            indexes = np.zeros(num_cluster * num_instances)
            for i in range(num_cluster):
                index = [i + k * num_cluster for k in range(num_instances)]
                samples = np.random.choice(np.where(pseudo_labels == i)[0], num_instances, True)
                indexes[index] = samples
            memory_features = features[indexes]
            return memory_features

        if epoch < args.intra_epoch:
             print("Strat intra-camera contrast learning")
             cam_cluster_features = []
             cam_text_cluster_features = []
             cam_memorys_features = []
             for c in np.unique(all_cams):
                 idx = np.where(all_cams == c)[0]
                 cam_label = torch.from_numpy(cam_labels[idx])
                 cam_features = features[idx]
                 cam_text_features = text_features[idx]

                 cam_cluster_features.append(generate_center_features(cam_label,cam_features))

                 cam_text_cluster_features.append(generate_center_features(cam_label, cam_text_features))

                 cam_memorys_features.append(
                     generate_instance_features(cam_label,
                                                cam_features,
                                                id_count_each_cam[c],
                                                args.num_instances))
                 del cam_label, cam_features


             cam_memorys = []
             for i in range(cameras):
                 cam_memorys.append(ClusterMemory(2048, id_count_each_cam[i], temp=args.temp,
                                                  momentum=args.momentum,num_instances=args.num_instances).cuda())

             for i, cam_memory in enumerate(cam_memorys):
                 if args.use_intra_hard:
                    cam_memory.features = F.normalize(
                             torch.cat(
                                 [cam_cluster_features[i],
                                        cam_memorys_features[i]], dim=0),
                        dim=1).cuda()
                 else:
                     cam_memory.features = F.normalize(cam_cluster_features[i],dim=1).cuda()


             trainer.cam_text_features = cam_text_cluster_features
             trainer.cam_memorys = cam_memorys
        # print("==> Initialize global centroid features in the hybrid memory")

        cluster_features = generate_center_features(pseudo_labels[global_labels], features)

        text_cluster_features = generate_center_features(pseudo_labels[global_labels], text_features)

        trainer.new_labels = pseudo_labels

        memory = ClusterMemory(2048, num_cluster, temp=args.temp,momentum=args.momentum).cuda()

        memory.features = F.normalize(cluster_features, dim=1).cuda()
        trainer.nums_class = num_cluster
        trainer.cluster_text_features = text_cluster_features

        trainer.memory = memory
        trainer.cam2accu = cam2accu
        trainer.pseudo2accu_mask = pseudo2accu_list
        train_loader = get_train_loader(loaders)

        curr_lr = optimizer.param_groups[0]['lr']
        print('=> Current Lr: {:.2e}'.format(curr_lr))

        trainer.train(epoch, train_loader, optimizer,optimizer_cc,
                      print_freq=args.print_freq,
                      train_iters = args.iters ,
                      camera = cameras ,
                      intra_epoch= args.intra_epoch,
                      )

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):

            print('=> Epoch {} test: '.format(epoch + 1))
            eval_results = test(model, test_loader)
            mAP = eval_results[0]
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP,  best_mAP)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'cam_classifier_state_dict': cam_classifier.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model.pth.tar'))

            print('rank1: {:4.1%}, rank5: {:4.1%}, rank10:{:4.1%} , mAP: {:4.1%}'.format(
                eval_results[1], eval_results[2], eval_results[3], eval_results[0]))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
        scheduler.step()
        torch.cuda.empty_cache()
        print('=> CUDA cache is released.')
        print('')

    print ('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    eval_results = test(model, test_loader)
    print('rank1: {:4.1%}, rank5: {:4.1%}, rank10:{:4.1%} , mAP: {:4.1%}'.format(
    eval_results[1], eval_results[2], eval_results[3], eval_results[0]))
    end_time = time.monotonic()
    dtime = timedelta(seconds=end_time - start_time)
    print('=> Task finished: {}'.format('CLIP_Stage2_ICS'))
    print('Stage2 running time: {}'.format(dtime))

