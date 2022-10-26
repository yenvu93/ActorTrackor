_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/movienet_detector.py', '../_base_/default_runtime.py'
]
USE_MMDET = True
model = dict(
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/home/yen/actortracker/model/cascade_rcnn_x101_64x4d_fpn.pth'  # noqa: E501
        ))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
total_epochs = 20
evaluation = dict(metric=['bbox'], interval=1)
