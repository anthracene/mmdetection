#_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
_base_ = '../detectors/cascade_rcnn_r50_sac_1x_coco.py'
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

#model=dict(roi_head=dict(bbox_head=dict(num_classes=2)))
model=dict(roi_head=dict(bbox_head=[dict(num_classes=2),dict(num_classes=2),dict(num_classes=2)]))
# model=dict(roi_head=dict(bbox_head=[
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=2,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=2,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=2,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))]))

evaluation = dict(interval=1, metric='mAP')
