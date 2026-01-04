# dataset configs
from sympy.logic.boolalg import distribute_xor_over_and

dataset_type = 'KizilDataset'
data_root = 'data/kizil'
# crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
tta_pipeline = []
train_dataloader = dict(  # Runner kwarg
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='labels/train'
        ),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(  # Runner kwarg
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='labels/val'
        ),
        pipeline=train_pipeline
    )
)
test_dataloader = dict(  # Runner kwarg
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='labels/test'
        ),
        pipeline=train_pipeline
    )
)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])  # Runner kwarg
test_evaluator = val_evaluator  # Runner kwarg

# model configs
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(  # Runner kwarg
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'  # noqa
model = dict(  # Runner arg
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
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
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),  # mit-b4
        embed_dims=64,  # mit-b4
        num_heads=[1, 2, 5, 8],  # mit-b4
        num_layers=[3, 8, 27, 3],  # mit-b4
    ),
    decode_head=dict(
        type='SegformerHead',
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
        in_channels=[64, 128, 320, 512],  # mit-b4
        channels=768,  # mit-b4
        num_classes=7,  # kizil
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# schedule configs
total_step = 100000
warmup_step = 1500
val_interval = 500
log_interval = 50
checkpoint_interval = 500
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(  # Runner kwarg
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
        }
    ),
)
# learning policy
param_scheduler = [  # Runner kwarg
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_step),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=warmup_step,
        end=total_step,
        by_epoch=False,
    )
]
# training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=total_step, val_interval=val_interval)  # Runner kwarg
val_cfg = dict(type='ValLoop')  # Runner kwarg
test_cfg = dict(type='TestLoop')  # Runner kwarg
default_hooks = dict(  # Runner kwarg
    checkpoint=dict(
        type='CheckpointHook',
        interval=checkpoint_interval,
        by_epoch=False,
        max_keep_ckpts=20,
        save_best=['mIoU', 'mDice'],
        rule='greater',
    ),
    logger=dict(
        type='LoggerHook',
        interval=log_interval,
        interval_exp_name=0,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    visualization=dict(type='SegVisualizationHook'),
)

# runtime configs
default_scope = 'mmseg'  # Runner kwarg
work_dir = 'runs/segformer'  # Runner arg
env_cfg = dict(  # Runner kwarg
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
visualizer = dict(  # Runner kwarg
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='mural-segmentation',
                name='segformer_mit-b4_4xb4-50k_kizil-512x512',
                notes="The dataset is made up of kizil_random_0104(and augmented), kizil_weighted_0104(and augmented), 11534 in total. "
                      "The dataset split ratio is 7:2:1, reduce_zero_label=False",
                resume='allow',
            ),
        )
    ],
    name='visualizer'
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

tta_model = dict(type='SegTTAModel')
