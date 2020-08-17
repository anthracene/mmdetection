_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
optimizer = dict(lr=0.02/2)

dataset_type = 'CustomDataset'
#data_root = '/mmdetection/data/IVC/'
data_root = '/sslow/XFORMED-test/'

total_epochs = 48

data = dict(
train = dict(
    type=dataset_type,
    ann_file=data_root + 'mmd-train.pkl',
    img_prefix=data_root + ''
    ),
val = dict(
    type=dataset_type,
    ann_file=data_root + 'mmd-test.pkl',
    img_prefix=data_root + ''
    ),
test = dict(
    #type=dataset_type,
    type='SubsetDataset',
    subset=[[0,0.05]],
    ann_file=data_root + 'mmd-test.pkl',
    img_prefix=data_root + '',
    # pipeline=[
    #     dict(type='LoadImageFromFile'),
    #     dict(type='LoadAnnotations', with_bbox=True),
    #     dict(
    #         type='MultiScaleFlipAug',
    #         img_scale=(1333, 800),
    #         flip=False,
    #         transforms=[
    #             dict(type='Resize', keep_ratio=True),
    #             dict(type='RandomFlip'),
    #             dict(
    #                 type='Normalize',
    #                 mean=[123.675, 116.28, 103.53],
    #                 std=[58.395, 57.12, 57.375],
    #                 to_rgb=True),
    #             dict(type='Pad', size_divisor=32),
    #             dict(type='ImageToTensor', keys=['img']),
    #             # dict(type='DefaultFormatBundleNoImg'),
    #             dict(type='Collect', keys=['img','gt_bboxes', 'gt_labels'])
    #         ])]
    # pipeline=[
    #     dict(type='LoadImageFromFile'),
    #     dict(type='LoadAnnotations', with_bbox=True),
    #     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    #     dict(type='RandomFlip', flip_ratio=0.5),
    #     dict(
    #         type='Normalize',
    #         mean=[123.675, 116.28, 103.53],
    #         std=[58.395, 57.12, 57.375],
    #         to_rgb=True),
    #     dict(type='Pad', size_divisor=32),
    #     dict(type='DefaultFormatBundle'),
    #     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]

)
)

model=dict(roi_head=dict(bbox_head=dict(num_classes=2)))

evaluation = dict(interval=1, metric='mAP')
