dataset_type = 'KizilDataset'
dataset_notes = (
    "The dataset is made up of kizil_random_0127(and augmented), kizil_weighted_0127(and augmented), 8838 in total."
    "The dataset split ratio is 7:1:2, KizilDataset.reduce_zero_label=True, *Loss.avg_non_ignore=True"
)
data_root = 'data/kizil'
crop_size = (512, 512)
dataset_classes = 6
data_preprocessor = dict(  # Runner kwarg
    type='SegDataPreProcessor',
    mean=[169.370, 158.813, 143.984],
    std=[46.466, 48.187, 47.635],
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
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
            seg_map_path='labels/train',
        ),
        pipeline=train_pipeline,
    ),
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
            seg_map_path='labels/val',
        ),
        pipeline=val_pipeline,
    ),
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
            seg_map_path='labels/test',
        ),
        pipeline=test_pipeline,
    ),
)
