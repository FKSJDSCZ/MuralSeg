_base_ = [
    '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/runtime/default_runtime.py',
]

# dataset
train_dataloader = dict(batch_size=8)
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
)
test_evaluator = val_evaluator

# model
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(  # Runner arg
    type='EncoderDecoder',
    data_preprocessor={{_base_.data_preprocessor}},
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        num_stages=4,
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),  # mit-b0
        embed_dims=32,  # mit-b0
        num_heads=[1, 2, 5, 8],  # mit-b0
        num_layers=[2, 2, 2, 2],  # mit-b0
    ),
    decode_head=dict(
        type='GHRMixFormerHead',
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True,
        ),
        in_channels=[32, 64, 160, 256],  # mit-b0
        channels=128,
        num_classes={{_base_.dataset_classes}},
        # head specified
        # attention
        pool_ratio=(1, 2, 4, 8),
        num_heads=(8, 5, 2, 1),
        attn_rank=0,
        attn_type='dual_route',
        dual_route_reduction_dim=(32, 32, 32, 32),
        dual_route_local_window=(3, 5, 5, 7),
        dual_route_global_pool=(32, 64, 128, 256),
        dual_route_use_kv_strip=True,
        dual_route_router_per_head=True,
        dual_route_router_mlp_ratio=0.25,
        dual_route_use_rel_pos_bias=True,
        with_strip_tokens=True,
        kv_gate=False,
        kv_gate_dim=[32, 64, 64, 64],
        # lpu
        with_lpu=True,
        lpu_dilations=(1, 3, 5),
        lpu_use_directional=True,
        lpu_directional_kernel=7,
        # ffn
        ffn_type='mlp',  # 'gated' or 'mlp'
        mlp_ratio=4,
        # elar
        with_elar=False,
        # drop
        drop=0,
        attn_drop=0,
        proj_drop=0,
        drop_path_rate=0.1,
        # fuse layer norm
        fuse_norm_cfg=norm_cfg,
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# schedule
schedule_total_step = 40000
schedule_warmup_step = 1500
schedule_val_interval = 500
schedule_log_interval = 50
schedule_checkpoint_interval = 500
schedule_max_lr = 6e-5
# optimizer
optim_wrapper = dict(  # Runner kwarg
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=schedule_max_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'rel_bias': dict(decay_mult=0.),
            'rel_pos_bias': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
        },
    ),
)
# learning policy
param_scheduler = [  # Runner kwarg
    dict(
        type='LinearLR',
        start_factor=1e-3,
        end=schedule_warmup_step,
        by_epoch=False,
    ),
    dict(
        type='PolyLR',
        eta_min=0.,
        power=1.,
        begin=schedule_warmup_step,
        end=schedule_total_step,
        by_epoch=False,
    ),
]
# training schedule
train_cfg = dict(  # Runner kwarg
    type='IterBasedTrainLoop',
    max_iters=schedule_total_step,
    val_interval=schedule_val_interval,
)
val_cfg = dict(type='ValLoop')  # Runner kwarg
test_cfg = dict(type='TestLoop')  # Runner kwarg
default_hooks = dict(  # Runner kwarg
    checkpoint=dict(
        type='MyCheckpointHook',
        interval=schedule_checkpoint_interval,
        by_epoch=False,
        max_keep_ckpts=1,
        save_best=['mIoU', 'mDice'],
        rule='greater',
    ),
    logger=dict(
        type='MyLoggerHook',
        interval=schedule_log_interval,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    visualization=dict(
        type='MySegVisualizationHook',
        interval=5,
    ),
)

# runtime
wandb_project = "muralseg-ablation"
wandb_notes = {{_base_.dataset_notes}}
visualizer = dict(  # Runner kwarg
    type='MySegLocalVisualizer',
    vis_backends=[
        dict(type='MyLocalVisBackend'),
        dict(
            type='MyWandbVisBackend',
            init_kwargs=dict(
                resume='allow',
            ),
        ),
    ],
    name='visualizer',
    alpha=0.5,
)
