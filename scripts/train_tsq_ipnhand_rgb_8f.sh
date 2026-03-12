

# train
export CUDA_VISIBLE_DEVICES=0,1
python main.py ipnhand RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 16 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=0.50 --shift_place=blockres --npb > ipnhand_dste_train.out 2>&1 