_base_ = '../cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py'

optimizer = dict(lr=0.02/2)

dataset_type = 'CustomDataset'
#data_root = '/mmdetection/data/IVC/'
data_root = '/sslow/XFORMED/'
ann_root = '/home/jmongan/Dev/IVCFilter/'

#total_epochs = 48

albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[-0.3, 0.3],
        contrast_limit=[-0.3, 0.3],
        brightness_by_max=False,
        p=0.0),
    dict(type='RandomRotate90', p=0.10)
]


data = dict(
train = dict(
    type=dataset_type,
    ann_file=ann_root + 'mmd-train.pkl',
    img_prefix=data_root + '',
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Albu',
            transforms=albu_train_transforms,
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_labels'],
                min_visibility=0.0,
                filter_lost_elements=True),
            keymap={
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            },
            update_pad_shape=False,
            skip_img_without_anno=False),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    ),
val = dict(
    type=dataset_type,
    ann_file=ann_root + 'mmd-val.pkl',
    img_prefix=data_root + ''
    ),
test = dict(
    #type=dataset_type,
    type='SubsetDataset',
    subset=[(0,1)],
    ann_file=ann_root + 'mmd-val.pkl',
    img_prefix=data_root + '',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1333, 800),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])
    ]

)
)

model=dict(roi_head=dict(bbox_head=['_merge_',dict(num_classes=1),dict(num_classes=1),dict(num_classes=1)]))

evaluation = dict(interval=1, metric='mAP')
#lr_config = dict(step=[8, 11])
