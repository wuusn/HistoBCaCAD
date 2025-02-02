_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/datasets/breast10x.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamW_10k.py'
]
num_classes = 2
model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True
        ),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=num_classes),
    auxiliary_head=dict(in_channels=512, num_classes=num_classes))

runner = dict(type='IterBasedRunner', max_iters=470000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)

data = dict(samples_per_gpu=8)
work_dir = 'data/Breast/split_L1_10x512_mask_Region9918_n2/save/one_swin_base_384'
load_from = 'checkpoints/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth'
