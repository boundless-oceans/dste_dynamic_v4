

# test 
export CUDA_VISIBLE_DEVICES=0,1
python test_models.py ipnhand \
    --weights=pretrained/ipnhand/TSQ_ipnhand_RGB_resnet50_shift0.50_blockres_avg_segment8_e50.pth \
    --test_segments=8 --test_crops=5 \
    --batch_size=4 -j 32  \
    --multi_clip_test > ipnhand_dete_test.out 2>&1