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
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa
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
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),  # mit-b3
        embed_dims=64,  # mit-b3
        num_heads=[1, 2, 5, 8],  # mit-b3
        num_layers=[3, 4, 18, 3],  # mit-b3
    ),
    decode_head=dict(
        type='DSFv2Head',
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
        in_channels=[64, 128, 320, 512],  # mit-b3
        channels=768,  # mit-b3
        num_classes={{_base_.dataset_classes}},
        corr_ratio=4,
        edge_dilations=(1, 3, 5),

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
schedule_max_lr = 1e-3
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
            'corr_self_weight': dict(decay_mult=0.),
            'gamma_edge': dict(decay_mult=0.),
            'gamma_smooth': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
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
        type='CosineAnnealingLR',
        eta_min=0.,
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
wandb_project = "mural-segmentation"
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
