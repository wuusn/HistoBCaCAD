#norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
num_classes=2
model = dict(
    type='EncoderDecoder',
    #pretrained='checkpoints/seg_cswin_large_384.pth',
    backbone=dict(
        #type='CSWinTransformer',
        type='CSWin',
        img_size=384,
        embed_dim=144,
        patch_size=4,
        depth=[2, 4, 32, 2],
        num_heads=[6,12,24,24],
        split_size=[1,2,12,12],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_chk=False,
        pretrained = '../../checkpoints/cswin_large_384.pth',
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[144, 288, 576, 1152],
        #in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=576,
        #in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        #loss_decode=dict(type='LovaszLoss', loss_name='loss_lovasz', reduction='none',loss_weight=0.4*0.6)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'Region'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    #dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    #dict(type='RandomFlip', prob=0.5, direction="vertical"),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    #dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = '/ssd/Breast/split_L1_10x512_mask_Region9918_n2'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_add_normal_out',
        ann_dir='train_add_normal_out',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir='val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='test',
        pipeline=test_pipeline))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=3e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)),
            head=dict(lr_mult=10.0)
            ))
#optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
max_iters=20000
runner = dict(type='IterBasedRunner', max_iters=max_iters)
#optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
checkpoint_config = dict(by_epoch=False, interval=max_iters)
evaluation = dict(interval=max_iters, metric='mIoU', pre_eval=True)
work_dir = './work_dirs/cswinl_fapn'
gpu_ids = range(0, 2)
