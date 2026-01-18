#!/bin/bash
python test.py \
    --root_path /home/ub/pql/Swin-Unet-main/datasets/Synapse/test_vol_h5 \
    --list_dir /home/ub/pql/Swin-Unet-main/lists/lists_Synapse \
    --num_classes 9 \
    --img_size 224 \
    --output_dir ./model_out \
    --max_epochs 400 \
    --batch_size 1 \
    --is_savenii \
    --test_save_dir ./predictions \
    --cfg /home/ub/pql/Swin-Unet-main/configs/swin_tiny_patch4_window7_224_lite.yaml \
    --resume /home/ub/pql/Swin-Unet-main/model_out/best_model.pth \
    --eval