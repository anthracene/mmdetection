_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
optimizer = dict(lr=0.02/2)

dataset_type = 'CustomDataset'
#data_root = '/mmdetection/data/IVC/'
data_root = '/sslow/XFORMED-test/'

#total_epochs = 48

img_norm_cfg = dict(
    mean=[116.93221962], std=[57.40954383], to_rgb=False) # luminance weighted average


train_pipeline=['_merge_',
                dict(color_type='grayscale'), # LoadImageFromFile
                None,
                None,
                None,
                dict(type='Normalize', **img_norm_cfg)
                ]
test_pipeline = ['_merge_',
    dict(type='LoadImageFromFile',color_type='grayscale'),
    dict(transforms=['_merge_',
                     None,
                     None,
                     dict(type='Normalize', **img_norm_cfg)
                     ])
]


data = dict(
train = dict(
    type=dataset_type,
    ann_file=data_root + 'mmd-train.pkl',
    img_prefix=data_root + '',
    pipeline=train_pipeline
    ),
val = dict(
    type=dataset_type,
    ann_file=data_root + 'mmd-test.pkl',
    img_prefix=data_root + '',
    pipeline=test_pipeline
    ),
test = dict(
    #type=dataset_type,
    type='SubsetDataset',
    subset=[(0,0.05)],
    ann_file=data_root + 'mmd-test.pkl',
    img_prefix=data_root + '',
    pipeline=test_pipeline
)
)

model=dict(
    pretrained='/sfast/mmdetection-checkpoints/gresnetxformed.pth',
    backbone=dict(in_channels=1),
    roi_head=dict(bbox_head=dict(num_classes=2)))

evaluation = dict(interval=1, metric='mAP')
