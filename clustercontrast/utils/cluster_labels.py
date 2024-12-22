import numpy as np
import torch
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from scipy.spatial.distance import cdist
from sklearn.cluster._dbscan import dbscan
from clustercontrast.evaluators import extract_vit_features


def img_association(args,model, propagate_loader,id_count_each_cam,rerank = False):
    print('Start Inference...')
    if args.dataset == 'market1501':
        associate_rate = 1.7 #TODO 1.9   Best 1.7

    elif args.dataset == 'msmt17':
        associate_rate = 1.3 #TODO msmt 1.5  rn 1.5  vit 1.3

    else:
        associate_rate = 1.50

    #or  associate_rate = 0.7 msmt market   or market 0.4
    print('  datsset:{},  associated: {}'.format(args.dataset,associate_rate))
    id_count_each_cam = np.sum(id_count_each_cam)

    get_features, features, ID_labels, all_cams, cam_labels, global_labels = extract_vit_features(model,propagate_loader,associate_rate)
    # compress intra-camera same-ID image features by average pooling
    new_features, new_cams, new_IDs = [], [], []
    for glab in np.unique(global_labels):
        idx = np.where(global_labels == glab)[0]
        new_features.append(np.mean(features[idx], axis=0))

        assert (len(np.unique(all_cams[idx])) == 1)
        assert (len(np.unique(ID_labels[idx])) == 1)
        new_cams.append(all_cams[idx[0]])
        new_IDs.append(ID_labels[idx[0]])

    new_features = np.array(new_features)
    new_cams = np.array(new_cams)
    new_IDs = np.array(new_IDs)
    del ID_labels, features

    # print("****",len(new_cams))
    # print("****",len(all_cams))
    # ** ** 3262
    # ** ** 12936

    # print("before cdist",new_features.shape) before cdist (3262, 2048)
    # compute distance and association
    new_features = new_features / np.linalg.norm(new_features, axis=1, keepdims=True)  # n * d -> n
    if rerank:
        W = compute_jaccard_distance(torch.from_numpy(new_features), k1=20, k2=6)
    else:
        W = cdist(new_features, new_features, 'sqeuclidean')
    # new_features = torch.from_numpy(new_features)
    # rerank_dist = compute_jaccard_distance(new_features, k1=30, k2=6)

    print('  distance matrix: shape= {}'.format(W.shape))  # should be (num_total_classes, num_total_classes)
    # self-similarity for association
    updated_label = propagate_label(W, new_IDs, new_cams, associate_rate * id_count_each_cam)

    print('  length of merged_label= {}, min= {}, max= {}'.format(len(updated_label), np.min(updated_label),
                                                                  np.max(updated_label)))

    # print("****",len(updated_label))
    return updated_label, get_features, global_labels, all_cams, cam_labels, new_IDs





def propagate_label(W, IDs, all_cams, associate_class_pair):
    # start label propagation
    print('Start associating ID...')

    associateMat = 1000 * np.ones(W.shape, W.dtype)

    # mask out intra-camera classes and lower-half
    for i in range(len(W)):
        W[i,np.where(all_cams==all_cams[i])[0]]=1000
        lower_ind=np.arange(0,i)
        W[i, lower_ind]=1000

    # cross-camera association
    associateMat=1000*np.ones(W.shape, W.dtype)

    # mask out intra-camera classes and lower-half
    for i in range(len(W)):
        W[i, np.where(all_cams == all_cams[i])[0]] = 1000
        lower_ind = np.arange(0, i)
        W[i, lower_ind] = 1000

    sorted_ind = np.argsort(W.flatten())[0:int(associate_class_pair)]
    row_ind = sorted_ind // W.shape[1]
    col_ind = sorted_ind % W.shape[1]

    C = len(np.unique(all_cams))
    cam_cover_info = np.zeros((len(W), C))
    associ_num, ignored_num = 0, 0
    associ_pos_num, ignored_pos_num = 0, 0
    #print('  associate_class_pair: {}'.format(associate_class_pair))
    thresh = associate_class_pair
    #print('  thresh= {}'.format(thresh))

    for m in range(len(row_ind)):
        cls1 = row_ind[m]
        cls2 = col_ind[m]  # cls1 -》cls2
        assert (all_cams[cls1] != all_cams[cls2])
        check = (cam_cover_info[cls1, all_cams[cls2]] == 0 and cam_cover_info[cls2, all_cams[cls1]] == 0)
        #
        if check:
            cam_cover_info[cls1, all_cams[cls2]] = 1
            cam_cover_info[cls2, all_cams[cls1]] = 1
            associateMat[cls1, cls2] = 0
            associateMat[cls2, cls1] = 0
            associ_num += 1
            if IDs[cls1] == IDs[cls2]:
                associ_pos_num += 1
        else:
            ignored_num += 1
            if IDs[cls1] == IDs[cls2]:
                ignored_pos_num += 1
        if associ_num >= thresh:
            break
    print('  associated class pairs: {}/{} correct, ignored class pairs: {}/{} correct'.
          format(associ_pos_num, associ_num, ignored_pos_num, ignored_num))

    # mask our diagnal elements
    for m in range(len(associateMat)):
        associateMat[m, m] = 0

    # Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1. oral = 3
    _, new_merged_label = dbscan(associateMat, eps=3, min_samples=2, metric='precomputed')
    # _, new_merged_label = dbscan(associateMat, eps=0.6, min_samples=4, metric='precomputed')
    # cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)



    del associateMat

    return new_merged_label



def img_association_sup(args,model, propagate_loader,id_count_each_cam,rerank = False):
    print('Start Inference...')
    if args.dataset == 'market1501':
        associate_rate = 1.5 #1.5 #market

    elif args.dataset == 'msmt17':
        associate_rate = 1.0 #msmt7 0.7DukeMTMC

    else:
        associate_rate = 1.0

    #or  associate_rate = 0.7 msmt market   or market 0.4
    print('  datsset:{},  associated: {}'.format(args.dataset,associate_rate))
    id_count_each_cam = np.sum(id_count_each_cam)

    get_features, features, ID_labels, all_cams, cam_labels, global_labels = extract_vit_features(model,propagate_loader,associate_rate)
    # compress intra-camera same-ID image features by average pooling
    new_features, new_cams, new_IDs = [], [], []
    for glab in np.unique(global_labels):
        idx = np.where(global_labels == glab)[0]
        new_features.append(np.mean(features[idx], axis=0))

        assert (len(np.unique(all_cams[idx])) == 1)
        assert (len(np.unique(ID_labels[idx])) == 1)
        new_cams.append(all_cams[idx[0]])
        new_IDs.append(ID_labels[idx[0]])

    new_features = np.array(new_features)
    new_cams = np.array(new_cams)
    new_IDs = np.array(new_IDs)
    del features
    # print("before cdist",new_features.shape) before cdist (3262, 2048)
    # compute distance and association

    # new_features = torch.from_numpy(new_features)
    # rerank_dist = compute_jaccard_distance(new_features, k1=30, k2=6)

    return new_IDs, get_features, global_labels, all_cams, cam_labels