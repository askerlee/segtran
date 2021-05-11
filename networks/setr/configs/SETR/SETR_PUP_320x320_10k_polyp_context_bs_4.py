_base_ = [
    '../_base_/models/setr_naive_pup.py',
    '../_base_/datasets/pascal_context.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(img_size=320, align_corners=False,
                  pos_embed_interp=True, drop_rate=0., norm_cfg=norm_cfg),
    decode_head=dict(img_size=320, align_corners=False, num_conv=4, upsampling_method='bilinear',
                     num_upsampe_layer=4, norm_cfg=norm_cfg, num_classes=2),
    auxiliary_head=[dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=9,
        img_size=320,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=14,
        img_size=320,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=19,
        img_size=320,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=23,
        img_size=320,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ])

optimizer = dict(lr=0.001, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

crop_size = (288, 288)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(288, 288))
find_unused_parameters = True
data = dict(samples_per_gpu=2)
