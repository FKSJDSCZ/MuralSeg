default_scope = 'mmseg'  # Runner kwarg
env_cfg = dict(  # Runner kwarg
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(  # Runner kwarg
    by_epoch=False,
    log_with_hierarchy=True,
)
log_level = 'INFO'  # Runner kwarg
load_from = None  # Runner kwarg
resume = False  # Runner kwarg
auto_scale_lr = None  # Runner kwarg
custom_hooks = None  # Runner kwarg
launcher = 'none'  # Runner kwarg
randomness = dict(  # Runner kwarg
    seed=3407,
    diff_rank_seed=False,
    deterministic=False,
)
experiment_name = None  # Runner kwarg
tta_model = None
