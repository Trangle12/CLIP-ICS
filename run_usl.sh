CUDA_VISIBLE_DEVICES=2,3 python train_clip.py --dataset 'market1501' --lossweight 0.6 --epsilon 0.8  --start_adv_epoch 40 \
--logs-dir '/data/CLIP-ICS-ReID/log/market1501'

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_clip.py --dataset 'msmt17' --lossweight 0.0 --epsilon 0.8  --start_adv_epoch 40 \
--logs-dir '/data/CLIP-ICS-ReID/log/msmt17'