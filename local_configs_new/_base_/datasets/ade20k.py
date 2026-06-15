dataset_type = 'ADE20KDataset'
dataset_notes = "ADEChallengeData2016"
data_root = 'data/ADEChallengeData2016'
crop_size = (512, 512)
dataset_classes = 150
data_preprocessor = dict(  # Runner kwarg
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],  # ImageNet
    std=[58.395, 57.12, 57.375],  # ImageNet
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,

)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        cat_max_ratio=0.75,
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(2048, 512),
        keep_ratio=True,
    ),
    # resize image to multiple of 32, improve SegFormer by 0.5-1.0 mIoU.
    dict(type='ResizeToMultiple', size_divisor=32),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')],
        ],
    ),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training',
        ),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation',
        ),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader
