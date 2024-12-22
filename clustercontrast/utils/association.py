import torch
import numpy as np
import copy
import collections
from collections import defaultdict
from collections import OrderedDict
import torch.nn.functional as F
import pdb

def association(cam2idx, rerank_dist, T=0.4):
    '''
    cam2idx 是一个字典 dict1, 它的 key1 是摄像头的编号 cam_id(以 market1501 数据集的训练集为例, cam_id 为 0 ~ 5), value1 也是
    一个字典 dict2, dict2 的 key2 是该摄像头下行人的编号 intra_id, (假设摄像头0号下面的行人类别有100个类别, 则 key2 为 0 ~ 99), 
    dict2 的 value2 是一个 list, list 中的元素为训练集中该摄像头下该类别行人的图像的全局编号(假设摄像头0号下面类别为0的行人, 他的
    图像有4张，则 cam2idx[0][0] = [0, 1, 2, 3], 摄像头0号下面类别为1的行人图像有5张, 则 cam2idx[0][1] = [4, 5, 6, 7, 8]。以 market1501
    数据集为例, list 中元素的取值为 0 ~ 12935) 
    (为什么list中编号是连续的？因为在训练集里他们也是连续的, 所以读取进来也是连续的，所以在下面的聚类算法里面就可以偷懒直接把距离矩阵
    按照这个编号来划分成小矩阵去求跨摄像头行人的最小相似度hhh, 即 line 41)

    rerank_dist 就是距离矩阵，以 makret1501 为例，它是一个 12936 * 12936 的矩阵，rerank_dist[i][j] 就是图像i和图像j之间的距离，
    这里的 i 和 j 都是图像在训练集中的全局编号，是由图像经过模型提取特征之后经过一个算法算出来的。

    T 是一个阈值，超参数，就是论文里 Section Ⅲ-C 中公式(5)里的 T。
    '''
    print('==> Start cross camera association')
    print('T:', T)
    edges=[]
    cam_num=len(list(cam2idx.keys()))
    for cam1 in range(cam_num):
        for cam2 in range(cam1+1, cam_num):
            for idx1 in range(len(list(cam2idx[cam1].keys()))):
                for idx2 in range(len(list(cam2idx[cam2].keys()))):
                    s1 = int(min(cam2idx[cam1][idx1]))         # 摄像头 cam1 下类别为 idx1 的行人图像的最小全局编号
                    e1 = int(max(cam2idx[cam1][idx1])) + 1     # 摄像头 cam1 下类别为 idx1 的行人图像的最大全局编号 + 1
                    s2 = int(min(cam2idx[cam2][idx2]))         # 摄像头 cam2 下类别为 idx2 的行人图像的最小全局编号
                    e2 = int(max(cam2idx[cam2][idx2])) + 1     # 摄像头 cam2 下类别为 idx2 的行人图像的最大全局编号 + 1
                    sub_dist = rerank_dist[s1:e1,s2:e2]        # 把 cam1 idx1 与 cam2 idx2 的行人图像距离矩阵截出来取最小值
                    min_dist = sub_dist.min()                  # 对应论文公式(4)
                    #min_dist = np.mean(sub_dist)
                    if min_dist < T:
                        edges.append((cam1,idx1,cam2,idx2,min_dist))   # 对应公式(5)，这一队跨摄像头行人之间存在一条边
    print("nums of edges:", len(edges))
    # 以下代码就是我们复现的 paper[17]的 greedy agglomeration algorithm，因为他们也没有开源，具体可以再去读一读他们的 paper
    groups = {}
    group_id = 0
    visiter = {}
    for cam in cam2idx.keys():
        if cam not in visiter.keys():
            visiter[cam] = {}
        for idx in cam2idx[cam].keys():
            visiter[cam][idx] = -1
    
    edges.sort(key=lambda x:x[4])
    for cam1,idx1,cam2,idx2,d in edges:
        if visiter[cam1][idx1] == -1 and visiter[cam2][idx2] == -1:
            groups[group_id]=[[cam1,idx1],[cam2,idx2]]
            visiter[cam1][idx1]=group_id
            visiter[cam2][idx2]=group_id
            group_id+=1
        elif visiter[cam1][idx1] != -1 and visiter[cam2][idx2] == -1:
            k=visiter[cam1][idx1]
            sub_groups=groups[k]
            sub_cams=[g[0] for g in sub_groups]
            if cam2 in sub_cams:
                continue
            else:
                visiter[cam2][idx2]=k
                groups[k].append([cam2,idx2])
        elif visiter[cam2][idx2] != -1 and visiter[cam1][idx1] == -1:
            k=visiter[cam2][idx2]
            sub_groups=groups[k]
            sub_cams=[g[0] for g in sub_groups]
            if cam1 in sub_cams:
                continue
            else:
                visiter[cam1][idx1]=k
                groups[k].append([cam1,idx1])
        elif visiter[cam2][idx2] != -1 and visiter[cam1][idx1] != -1:
            k1=visiter[cam1][idx1]
            k2=visiter[cam2][idx2]
            sub_groups1=groups[k1]
            sub_cams1=[g[0] for g in sub_groups1]
            sub_groups2=groups[k2]
            sub_cams2=[g[0] for g in sub_groups2]
            if len(set(sub_cams1)&set(sub_cams2))==0:
                new_groups=[]
                new_groups.extend(sub_groups1)
                new_groups.extend(sub_groups2)
                groups[group_id]=new_groups
                for g in groups[group_id]:
                    visiter[g[0]][g[1]]=group_id
                group_id+=1
                groups.pop(k1)
                groups.pop(k2)
    
    
    newgroups = {}
    for i,key in enumerate(groups.keys()):
        newgroups[i] = groups[key]
    
    return newgroups

# new_gropus {0: [[0, 123], [1, 113], [2, 122], [4, 394]],
#             1: [[0, 307], [1, 253], [2, 305]],
#             2: [[0, 445], [1, 325], [2, 570], [4, 554]],
#             3: [[0, 39], [2, 35], [3, 26], [1, 197]],
#             4: [[0, 183], [2, 360], [1, 289], [5, 261], [4, 161]],
#             5: [[0, 457], [2, 473], [1, 372], [3, 91], [4, 405]],

    # cam2c_p_label = [[x,y] for x,y in zip(all_cams,cam_labels)]
    #
    # cam2idx = {}
    # dataset_len = list(range(0,len(global_labels),1))
    # updated_label = [-1 for _ in range(len(global_labels))]
    # for i in range(len(all_cams)):
    #     cam_id = all_cams[i]
    #     intra_id = cam_labels[i]
    #     global_idx = dataset_len[i]
    #
    #     if cam_id not in cam2idx:
    #         cam2idx[cam_id] = {}
    #     if intra_id not in cam2idx[cam_id]:
    #         cam2idx[cam_id][intra_id] = []
    #
    #     cam2idx[cam_id][intra_id].append(global_idx)



    # rerank_dist = compute_jaccard_distance(get_features, k1=30, k2=6)
    #
    # new_grops = association(cam2idx,rerank_dist)
    #
    # for key,value in new_grops.items():
    #     indices = [cam2c_p_label.index(item) for item in value if item in cam2c_p_label ]
    #
    #     for i in list(indices):
    #         updated_label[i] = key
    #
    #
    # num_clusters = len(set(updated_label)) - (1 if -1 in updated_label else 0)