import os
import torch
import torch.nn as nn
from torch.cuda import amp

import tqdm

from clustercontrast.losses.supcon import SupConLoss
from clustercontrast.utils.meters import AverageMeter


def do_train_stage1(args,
             model,
             loaders_oral,
             optimizer,
             scheduler,
             ):
    checkpoint_period = args.epochs_stage1
    device = "cuda"
    epochs = args.epochs_stage1

    print('start training')
    if args.dataset == 'market1501':
        all_cams =  [0, 1, 2, 3, 4, 5]
        log_period = 100
    elif args.dataset == 'msmt17':
        all_cams = [0, 1, 2, 3, 4, 5, 6,7 , 8, 9, 10, 11, 12, 13, 14]
        log_period = 200
    else:
        all_cams = [0, 1, 2, 3, 4, 5 , 6, 7]
        log_period = 100

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    image_features_oral = []
    g_labels_list_oral = []
    cam_list = []
    ######################  ICS  with global labels   #################################
    # with torch.no_grad():
    #     for c, (imgs, _, cams, _, cam_label, g_label, _) in enumerate(tqdm.tqdm(loaders_oral.propagate_loader)):
    #         imgs = imgs.to(device)
    #         g_label = g_label.to(device)
    #         with amp.autocast(enabled=True):
    #             image_feature = model(imgs, g_label, get_image=True)
    #             for i, img_feat in zip(g_label, image_feature):
    #                 g_labels_list_oral.append(i)
    #                 image_features_oral.append(img_feat.cpu())
    #
    #     g_labels_list = torch.stack(g_labels_list_oral, dim=0).cuda()
    #     image_features_list = torch.stack(image_features_oral, dim=0).cuda()
    #######################  ICS  with cams labels   ##################################
    with torch.no_grad():
        for c, (imgs, _, cams, _, cam_label, g_label, _) in enumerate(tqdm.tqdm(loaders_oral.propagate_loader)):
            imgs = imgs.to(device)
            g_label = g_label.to(device)
            cams = cams.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(imgs, g_label, get_image=True)

                for i, img_feat,cam in zip(g_label, image_feature , cams):
                    g_labels_list_oral.append(i)
                    image_features_oral.append(img_feat.cpu())
                    cam_list.append(cam)

        g_labels_list = torch.stack(g_labels_list_oral, dim=0).cuda()
        cam_label_list = torch.stack(cam_list, dim=0).cuda()
        image_features_list = torch.stack(image_features_oral, dim=0).cuda()

        batch = 64
        num_image = g_labels_list.shape[0]

        print("num_images = {}".format(num_image))
        i_ter = num_image // batch

    del g_labels_list_oral, image_features_oral
    ###############################################################
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]
            target = g_labels_list[b_list]
            cams = cam_label_list[b_list]
            image_features = image_features_list[b_list]
            loss = 0
            with amp.autocast(enabled=True):
                text_features = model(label=target, get_text=True)
                for ii in range(len(all_cams)):  # all_cams=[0,1,2,3,4,5], make sure all_cams contains all actual cam values
                    target_cam = all_cams[ii]
                    if torch.nonzero(cams == target_cam).size(0) > 0:
                        percam_feat = image_features[cams == target_cam]
                        percam_text_feat = text_features[cams == target_cam]
                        percam_label = target[cams == target_cam]
                        # print("percam_feat.shape = {}".format(percam_feat.shape))
                        # print("percam_text_feat.shape = {}".format(percam_text_feat.shape))
                        #print("percam_label.shape = {}".format(percam_label.shape))

                        loss_i2t = xent(percam_feat, percam_text_feat, percam_label, percam_label)

                        loss += loss_i2t

                        loss_t2i = xent(percam_text_feat, percam_feat, percam_label, percam_label)

                        loss += loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item())

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                    print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                              .format(epoch, (i + 1), i_ter + 1,
                                      loss_meter.avg, scheduler._get_lr(epoch)[0]))

        OUTPUT_DIR = args.logs_dir
        MODEL_NAME = 'RN50'
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, MODEL_NAME + '_stage1_{}.pth'.format(epoch)))


    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    print('=> Task finished: {}'.format('CLIP-ICS-Stage1'))
    print("Stage1 running time: {}".format(total_time))
