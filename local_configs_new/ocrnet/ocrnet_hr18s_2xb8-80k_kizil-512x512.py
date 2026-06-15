_base_ = [
    '../_base_/datasets/kizil.py',
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
checkpoint = 'open-mmlab://msra/hrnetv2_w18_small'
model = dict(  # Runner arg
    type='CascadeEncoderDecoder',
    data_preprocessor={{_base_.data_preprocessor}},
    num_stages=2,
    pretrained=checkpoint,
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(2, 2),
                num_channels=(18, 36),
            ),
            stage3=dict(
                num_modules=3,
                num_branches=3,
                block='BASIC',
                num_blocks=(2, 2, 2),
                num_channels=(18, 36, 72),
            ),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='BASIC',
                num_blocks=(2, 2, 2, 2),
                num_channels=(18, 36, 72, 144),
            ),
        ),
    ),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[18, 36, 72, 144],
            channels=sum([18, 36, 72, 144]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes={{_base_.dataset_classes}},
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4,
                avg_non_ignore=True,
            ),
        ),
        dict(
            type='OCRHead',
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes={{_base_.dataset_classes}},
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                avg_non_ignore=True,
            ),
        ),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# schedule
schedule_total_step = 80000
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
