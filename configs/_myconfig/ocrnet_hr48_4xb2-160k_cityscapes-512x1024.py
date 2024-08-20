_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=4,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=4,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])

# dataset settings
dataset_type = 'CityscapesDatasetCrack'
data_root = '/home/veily/1sjq/DataSet/RoadImg/road_crack_gray/'
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root))
test_dataloader = val_dataloader