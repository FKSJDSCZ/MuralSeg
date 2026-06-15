_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/runtime/default_runtime.py',
]

# dataset
train_dataloader = dict(batch_size=4)
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
        type='EHCAHead',
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
        num_heads=(8, 5, 2, 1),  # s4 -> s1
        pool_ratio=(1, 2, 4, 8),  # s4 -> s1
        # block hyperparams
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        qkv_bias=True,
        # hybrid interaction switches
        use_cross=True,
        use_agent=True,
        # cross position bias
        use_cross_pos_bias=True,
        cross_pos_bias_base_size=7,
        # agent relative bias
        use_agent_pos_bias=True,
        agent_pos_bias_base_size=7,
        agent_pos_bias_base_window_size=7,
        # agent shapes & edge agents per stage
        agent_shape=((7, 7), (7, 7), (7, 7), (7, 7)),
        use_edge_agent=(True, True, True, True),
        use_agent_dwc=True,
        agent_dwc_kernel_size=3,

    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# schedule
schedule_total_step = 160000
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
            'head': dict(lr_mult=10.),
        },
    ),
)
# learning policy
param_scheduler = [  # Runner kwarg
    dict(
        type='LinearLR',
        start_factor=1e-6,
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
randomness = dict(
    seed=114514,
)
