dataset_type = 'MuralDHDataset'
dataset_notes = (
    "The dataset is made up of MuralDH0124(and augmented). "
    "MuralDHDataset.reduce_zero_label=False, *Loss.avg_non_ignore=True, *Head.ignore_index=255."
)
data_root = 'data/MuralDH'
crop_size = (512, 512)
dataset_classes = 2
data_preprocessor = dict(  # Runner kwarg
    type='SegDataPreProcessor',
    mean=[120.269, 112.794, 86.699],
    std=[55.811, 54.815, 51.741],
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
    batch_size=1,
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
            img_path='images/test',
            seg_map_path='labels/test',
        ),
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader