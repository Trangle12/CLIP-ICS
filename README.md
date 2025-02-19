<p align="center">
  <h1 align="center">CLIP-based Camera-Agnostic Feature Learning for Intra-camera Supervised Person Re-Identification</h1>
  <p align="center">
    <a href="https://scholar.google.com.tw/citations?hl=zh-CN&user=3jWe9KAAAAAJ" rel="external nofollow noopener" target="_blank"><strong>Xuan Tan</strong></a>
    ·
    <a href="https://scholar.google.com.tw/citations?user=7PqgUw4AAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Xun Gong</strong></a>
    ·
    <a href="https://scholar.google.com.tw/citations?user=oW4vMVkAAAAJ&hl=zh-CN&oi=sra" target="_blank"><strong>Yang Xiang</strong></a>
  </p>
<p align="center">
 <a href="https://arxiv.org/abs/2409.19563" rel="external nofollow noopener" target="_blank">Arxiv Paper Link</a>
<p align="center">
 <a href="https://ieeexplore.ieee.org/document/10813454" rel="external nofollow noopener" target="_blank">IEEE Paper Link</a>

  
### Method
![CCAFL](imgs/overview.png)

This is an official code implementation of "CLIP-based Camera-Agnostic Feature Learning for Intra-camera Supervised Person Re-Identification".

### Update Time
-- 2024-12 The code has released!!!
 
-- 2024-12 CCAFL is accepted by TCSVT25! 🎉🎉

-- 2024-9 We will release the code when the paper is accepted.

### Preparation

Download the datasets:

For privacy reasons, we don't have the dataset's copyright. Please contact authors to get this dataset.

```

Market-1501-v15.09.15/
├── bounding_box_test
├── bounding_box_train
├── gt_bbox
├── gt_query
└── query

MSMT17/
├── bounding_box_test
├── bounding_box_train
└── query

DukeMTMC-reID/
├── bounding_box_test
├── bounding_box_train
└── query

```

### Installation

```
conda create -n ccafl python=3.9
conda activate ccafl
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install tqdm
conda install scikit-learn

pip install yacs
pip install timm
pip install scikit-image
pip install ftfy
pip install easydict
pip install regex
pip install faiss-gpu
```


## Training
```shell
sh run_usl.sh 
```

## Performance

![perf](imgs/performance.png)

### Intra-camera Supervised Person ReID

##### Market-1501
| Model        |Pretrained	 | Image Size|Paper | Download |
| :------:     |  :------: | :------: |:------: |:------: |
| Resnet50     | CLIP | 256*128 |90.1/96.0 |[model](https://drive.google.com/file/d/1-o7XVkheqhIUV_QUayqmp0goO87xSf4Y/view?usp=drive_link) / [log](https://drive.google.com/file/d/1TeoGPORL3HjzHIejbNH4h-8lHk8FaXwE/view?usp=drive_link)|

##### MSMT17
| Model      |Pretrained  | Image Size|Paper | Download |
| :------:     |  :------: |  :------: |:------: |:------: |
| Resnet50    | CLIP | 256*128 |58.9/82.9 |[model](https://drive.google.com/file/d/1EI5Bv9Y_bZISW9Cql7-VXIuspfPtcEwc/view?usp=drive_link) / [log](https://drive.google.com/file/d/1e_aOp7RJd4Zc3cKn_VfmEFZ9Y7SnazLT/view?usp=drive_link)|

##### DukeMTMC-ReID
| Model     |Pretrained    | Image Size|Paper | Download |
| :------:    |  :------:    | :------: |:------: |:------: |
| Resnet50    | CLIP | 256*128 |81.5/90.8 |[model](https://drive.google.com/file/d/1c_bJlIe42ByHoKBkhNDMfZ56bNv5R3cu/view?usp=drive_link) / [log](https://drive.google.com/file/d/1tpXkMR5xEwZ-sAXk_7Tard7bfXqBGbKv/view?usp=drive_link)|

## Citation
If our work is helpful for your research, please consider citing:
```bibtex
@ARTICLE{10534060,
  author={Gong, Xun and Tan, Xuan and Xiang, Yang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Contrastive Mean Teacher for Intra-camera Supervised Person Re-Identification}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Cameras;Pedestrians;Training;Feature extraction;Computational modeling;Lighting;Data models;Intra-camera supervision;Mean Teacher;Contrastive learning;Person re-identification},
  doi={10.1109/TCSVT.2024.3402533}}

@ARTICLE{10813454,
  author={Tan, Xuan and Gong, Xun and Xiang, Yang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={CLIP-based Camera-Agnostic Feature Learning for Intra-camera Supervised Person Re-Identification}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Cameras;Pedestrians;Annotations;Identification of persons;Semantics;Representation learning;Contrastive learning;Accuracy;Training;Labeling;Person re-identification;CLIP;intra-camera supervision;camera-based adversarial loss},
  doi={10.1109/TCSVT.2024.3522178}}

```

### References.

[1] Bianchi, Federico, et al. "Contrastive language-image pre-training for the italian language." arXiv preprint arXiv:2108.08688 (2021).

[2] Zhou, Kaiyang, et al. "Learning to prompt for vision-language models." International Journal of Computer Vision 130.9 (2022): 2337-2348.

[3] Li, Siyuan, Li Sun, and Qingli Li. "Clip-reid: Exploiting vision-language model for image re-identification without concrete text labels." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 1. 2023.

[4] Dai, Zuozhuo, et al. "Cluster contrast for unsupervised person re-identification." Proceedings of the Asian Conference on Computer Vision. 2022.


