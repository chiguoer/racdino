import torch
pi = torch.pi

dataset_type = 'CustomNuScenesDataset_radar'
dataset_root = '/data/dataset/RacFormer/nuscenes/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True
)

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_range = [-51.2, -51.2, 51.2, 51.2]  # 新增BEV范围参数
voxel_size = [0.2, 0.2, 8]
radar_voxel_size = [0.8, 0.8, 8]  # 雷达体素大小
radar_use_dims = [0, 1, 2, 8, 9, 18]  # 雷达使用的维度

embed_dims = 384  # 适配DINOv2的嵌入维度
num_layers = 6

num_frames = 8
num_levels = 4
num_points = 4
num_points_bev = 4
img_depth_num = 3
bev_depth_num = 5 
d_region_list = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

num_clusters = 6
num_ray = 150
num_query = num_ray * num_clusters

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (224, 448),  # 适配DINOv2输入尺寸
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 65.0, 96.0],
    'rcs': [-64, 64, 64]
}

numC_Trans = 256
file_client_args = dict(backend='disk')

# 替换为DINOv2骨干网络
img_backbone = dict(
    type='DinoAdapter',
    add_vit_feature=False,
    pretrain_size=518,
    pretrained_vit=True,
    num_heads=6,
    embed_dim=embed_dims,
    freeze_dino=True,  # 冻结DINOv2预训练权重
)

# 调整neck适配DINOv2输出
img_neck = dict(
    type='FPN',
    in_channels=[384, 384, 384, 384],  # 匹配DINOv2输出通道
    out_channels=embed_dims,
    num_outs=num_levels)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # 统一为DINOv2的归一化参数
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

img_lss_neck=dict(
    type='CustomFPN',
    in_channels=[384, 384],  # 适配DINOv2输出
    out_channels=256,
    num_outs=1,
    start_level=0,
    out_ids=[0])

img_lss_view_transformer=dict(
    type='LSSViewTransformerBEVDepth_racformer',
    grid_config=grid_config,
    input_size=ida_aug_conf['final_dim'],
    in_channels=256,
    out_channels=numC_Trans,
    depthnet_cfg=dict(use_dcn=False),
    downsample=16,
    loss_depth_weight=2.0)

pre_process=None
model = dict(
    type='RaCFormer',
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=14)),  # 适配DINOv2的pad参数
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    img_lss_neck=img_lss_neck,
    img_lss_view_transformer=img_lss_view_transformer,
    num_lss_fpn=2,
    dep_downsample=16,
    pre_process=pre_process,
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=radar_voxel_size,  # 使用雷达体素尺寸
        max_voxels=(30000, 40000),
        point_cloud_range=point_cloud_range,
        deterministic=False,), 

    radar_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=len(radar_use_dims),  # 匹配雷达使用维度
        feat_channels=[64],
        with_distance=False,
        voxel_size=radar_voxel_size,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),

    radar_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(128, 128)),

    pts_bbox_head=dict(
        type='RaCFormer_head',
        num_classes=10,
        num_clusters=num_clusters,
        in_channels=embed_dims,
        num_query=num_query,
        query_denoising=True,
        query_denoising_groups=10,
        code_size=10,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        transformer=dict(
            type='RaCFormerTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_points_bev=num_points_bev,
            img_depth_num=img_depth_num, 
            bev_depth_num=bev_depth_num,
            num_layers=num_layers,
            num_levels=num_levels,
            num_ray=num_ray,
            num_classes=10,
            code_size=10,
            pc_range=point_cloud_range,
            d_region_list=d_region_list),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.05,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='PolarHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            theta_cost=dict(type='ThetaL1Cost', weight=3.0),
            iou_cost=dict(type='IoUCost', weight=0.0),
        )
    ))
)


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),  # 适配DINOv2的float32输入
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False,
         with_label=True, with_bbox_depth=True),  # 新增2D标注加载
    # 雷达数据加载替换为DINOv2的处理方式
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=6,
        use_num=6,
        use_dim=radar_use_dims,
        max_num=2048),
    dict(type='RadarRangeFilter', radar_range=bev_range),  # 雷达范围过滤
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),  # 替换图像变换
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],  # 关闭平移增强
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True),
    dict(type='NormalizeMultiviewImage',** img_norm_cfg),  # 替换归一化方式
    dict(type='PadMultiViewImage', size_divisor=14),  # 适配DINOv2的pad参数
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='RadarPointToMultiViewDepth', downsample=1, grid_config=grid_config, test_mode=False),
    dict(type='RaCFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', 
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_depth', 'radar_depth', 'radar_rcs', 'radar_points', 'gt_bboxes', 'gt_labels'], 
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'intrinsics', 'scene_token'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=6,
        use_num=6,
        use_dim=radar_use_dims,
        max_num=2048),
    dict(type='RadarRangeFilter', radar_range=bev_range),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='RadarPointToMultiViewDepth', downsample=1, grid_config=grid_config, test_mode=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RaCFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'gt_depth', 'radar_points', 'radar_depth', 'radar_rcs'], 
                 meta_keys=('filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                            'lidar2img', 'img_timestamp', 'intrinsics'))
        ])
]

data = dict(
    workers_per_gpu=6,  # 调整工作进程数
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        # 新增DINOv2的时序训练参数
        num_frame_losses=1,
        seq_split_num=2,
        seq_mode=True,
        queue_length=1),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        queue_length=1),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        queue_length=1),
    # 新增采样器配置
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,  # 调整学习率适配DINOv2
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),  # DINOv2骨干使用较小学习率
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

# 替换为梯度累积优化器
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook',
    loss_scale='dynamic',
    cumulative_iters=16,  # 梯度累积步数
    grad_clip=dict(max_norm=35, norm_type=2)
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 90  # 延长训练周期适配DINOv2
batch_size = 1  # 调整batch size

load_from = None  # DINOv2使用自带预训练权重
revise_keys = [('backbone', 'img_backbone')]

resume_from = None

default_hooks = dict(
    checkpoint = None
)

checkpoint_config = dict(interval=1001, max_keep_ckpts=3)  # 调整 checkpoint 间隔

log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', interval=50, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ]
)

eval_config = dict(interval=2)

debug = False

# 新增DINOv2的自定义钩子
custom_hooks = [
    dict(
        type='SequentialControlHook',
        start_epoch=18,
    ),
    dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL'),
    dict(
        type='CheckInvalidLossHook',
        interval=1,
        priority='VERY_HIGH'
    )
]
