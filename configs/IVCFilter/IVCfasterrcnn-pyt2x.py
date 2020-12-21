_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'

optimizer = dict(lr=0.02/2)

dataset_type = 'CustomDataset'
#data_root = '/mmdetection/data/IVC/'
data_root = '/sslow/XFORMED/'
ann_root = '/home/jmongan/Dev/IVCFilter/'

#total_epochs = 48

data = dict(
train = dict(
    type=dataset_type,
    ann_file=ann_root + 'mmd-train.pkl',
    img_prefix=data_root + ''
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

model=dict(roi_head=dict(bbox_head=dict(num_classes=1)))

evaluation = dict(interval=1, metric='mAP')
#lr_config = dict(step=[8, 11])
