python tools/test.py \
    local_configs_new/segformer/segformer_mit-b0_2xb8-80k_kizil-512x512.py \
    runs/trains/segformer_mit-b0_2xb8-80k_kizil-512x512/20260210_053822/best_mIoU_iter_75500.pth \
    --draw
python tools/test.py \
    local_configs_new/segnext/segnext_mscan-t_2xb8-80k_kizil-512x512.py \
    runs/trains/segnext_mscan-t_2xb8-80k_kizil-512x512/20260210_052705/best_mIoU_iter_69500.pth \
    --draw
python tools/test.py \
    local_configs_new/feedformer/feedformer_mit-b0_2xb8-80k_kizil-512x512.py \
    runs/trains/feedformer_mit-b0_2xb8-80k_kizil-512x512/20260210_043435/best_mIoU_iter_70000.pth \
    --draw
python tools/test.py \
    local_configs_new/edaformer/edaformer_eft-t_2xb8-80k_kizil-512x512.py \
    runs/trains/edaformer_eft-t_2xb8-80k_kizil-512x512/20260210_173410/best_mIoU_iter_67500.pth \
    --draw
python tools/test.py \
    local_configs_new/umixformer/umixformer_mit-b0_2xb8-80k_kizil-512x512.py \
    runs/trains/umixformer_mit-b0_2xb8-80k_kizil-512x512/20260207_050944/best_mIoU_iter_67500.pth \
    --draw
python tools/test.py \
    local_configs_new/offseg/offseg_efficientformerv2-s1_2xb8-80k_kizil-512x512.py \
    runs/trains/offseg_efficientformerv2-s1_2xb8-80k_kizil-512x512/20260212_204059/best_mIoU_iter_72000.pth \
    --draw