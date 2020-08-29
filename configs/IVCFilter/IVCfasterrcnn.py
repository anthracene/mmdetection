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
    subset=[(0,0.05)],
    ann_file=data_root + 'mmd-test.pkl',
    img_prefix=data_root + ''
)
)

model=dict(roi_head=dict(bbox_head=dict(num_classes=2)))

evaluation = dict(interval=1, metric='mAP')
