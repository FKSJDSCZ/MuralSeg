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
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
checkpoint = 'pretrained/efficientformerv2-s1.pth'
model = dict(  # Runner arg
    type='EncoderDecoder',
    data_preprocessor={{_base_.data_preprocessor}},
    pretrained=None,
    backbone=dict(
        type='efficientformerv2_s1_feat',
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    ),
    decode_head=dict(
        type='OffSegHead',
        in_channels=[32, 48, 120, 224],
        new_channels=[32, 32, 64, 128],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes={{_base_.dataset_classes}},
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(480, 480)),
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
