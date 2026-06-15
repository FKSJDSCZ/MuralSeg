#image="data/cityscapes/leftImg8bit/train/bochum/bochum_000000_000313_leftImg8bit.png"
image="data/cityscapes/leftImg8bit/train/dusseldorf/dusseldorf_000000_000019_leftImg8bit.png"
config="local_configs_new/biagent/mit-b0_biagent-hy-interp-bid-q-ddpu-dwc-out128_4xb4-160k_cityscapes-1024x1024.py"
checkpoint="runs/trains/mit-b0_biagent-hy-interp-bid-q-ddpu-dwc-out128_4xb4-160k_cityscapes-1024x1024/20260311_124228/best_mIoU_iter_144000.pth"
category_index="9"

python tools/analysis_tools/visualization_cam.py ${image} ${config} ${checkpoint} \
    --out-file vis/feature_map/prediction.png \
    --cam-file vis/feature_map/decoder_s1_forward_attn_cls${category_index}.png \
    --target-layers decode_head.blocks[3].mlp1 \
    --category-index ${category_index}

python tools/analysis_tools/visualization_cam.py ${image} ${config} ${checkpoint} \
    --out-file vis/feature_map/prediction.png \
    --cam-file vis/feature_map/decoder_s1_feedback_attn_cls${category_index}.png \
    --target-layers decode_head.blocks[3].mlp2 \
    --category-index ${category_index}

python tools/analysis_tools/visualization_cam.py ${image} ${config} ${checkpoint} \
    --out-file vis/feature_map/prediction.png \
    --cam-file vis/feature_map/decoder_s1_cls${category_index}.png \
    --target-layers decode_head.blocks[3] \
    --category-index ${category_index}