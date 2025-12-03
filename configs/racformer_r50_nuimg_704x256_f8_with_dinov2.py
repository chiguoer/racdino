# RaCFormer配置文件 - 集成DINOv2语义增强
# 基于racformer_r50_nuimg_704x256_f8.py，添加dinov2_adapter模块

import torch
pi = torch.pi

dataset_type = 'CustomNuScenesDataset_radar'
#dataset_root = 'data/nuscenes/'
dataset_root = '/data/dataset/RacFormer/nuscenes/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True
)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# arch config
embed_dims = 256
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
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 65.0, 96.0],
    'rcs': [-64, 64, 64]
}

numC_Trans = 256
file_client_args = dict(backend='disk')

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)

# ========== DINOv2 Adapter 配置 ==========
# 用于在ResNet编码后增强图像语义特征
dinov2_adapter = dict(
    type='DinoAdapter',
    num_heads=12,              # ViT-Base: 12, ViT-Small: 6
    embed_dim=768,             # ViT-Base: 768, ViT-Small: 384
    depth=12,                  # Transformer层数：ViT-Base: 12, ViT-Small: 12
    pretrain_size=518,         # DINOv2预训练图像尺寸
    pretrained_vit=True,       # 加载DINOv2预训练权重
    freeze_dino=True,          # 冻结DINOv2参数（推荐，节省显存）
    patch_size=14,             # DINOv2 patch大小
    mlp_ratio=4,
    conv_inplane=64,
    n_points=4,
    deform_num_heads=6,
    init_values=0.,
    interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
    with_cffn=True,
    cffn_ratio=0.25,
    deform_ratio=1.0,
    add_vit_feature=False,
    use_extra_extractor=True,
    with_cp=False,             # 如果显存不足，可设为True启用gradient checkpointing
)

img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

img_lss_neck=dict(
    type='CustomFPN',
    in_channels=[1024, 2048],
    out_channels=256,
    num_outs=1,
    start_level=0,
    out_ids=[0])

img_lss_view_transformer=dict(
    type='LSSViewTransformerBEVDepth_racformer',
    grid_config=grid_config,
    input_size=ida_aug_conf['final_dim'],
    in_channels=256,
    out_channels=256,
    downsample=16)

pre_process = dict(
    type='CustomResNet',
    numC_input=256,
    num_layer=[2, ],
    num_channels=[numC_Trans, ],
    stride=[1, ],
    backbone_output_ids=[0, ])

# radar point cloud range
radar_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
radar_voxel_size = [0.8, 0.8, 8]
radar_voxel_layer = dict(
    max_num_points=10,  # max_points_per_voxel
    point_cloud_range=radar_point_cloud_range,
    voxel_size=radar_voxel_size,
    max_voxels=(30000, 40000))
radar_voxel_encoder = dict(
    type='HardSimpleVFE',
    num_features=4,
)
radar_middle_encoder = dict(
    type='PointPillarsScatter',
    in_channels=4,
    output_shape=[128, 128])

# ========== 主模型配置 ==========
model = dict(
    type='RaCFormer',
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    
    # ========== 添加DINOv2 Adapter ==========
    dinov2_adapter=dinov2_adapter,
    
    img_backbone=img_backbone,
    img_neck=img_neck,
    num_lss_fpn=2,
    dep_downsample=16,
    img_lss_neck=img_lss_neck,
    img_lss_view_transformer=img_lss_view_transformer,
    pre_process=pre_process,
    radar_voxel_layer=radar_voxel_layer,
    radar_voxel_encoder=radar_voxel_encoder,
    radar_middle_encoder=radar_middle_encoder,
    
    pts_bbox_head=dict(
        type='RaCFormer_head',
        embed_dims=embed_dims,
        num_query=num_query,
        num_clusters=num_clusters,
        num_classes=len(class_names),
        in_channels=embed_dims,
        code_size=10,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        transformer=dict(
            type='RaCFormerTransformer',
            num_frames=num_frames,
            num_layers=num_layers,
            d_region_list=d_region_list,
            embed_dims=embed_dims,
            num_points=num_points,
            num_points_bev=num_points_bev,
            num_levels=num_levels,
            img_depth_num=img_depth_num,
            bev_depth_num=bev_depth_num,
            pc_range=point_cloud_range,
            num_groups=4,
            decoder=dict(
                type='RaCFormerTransformerDecoderLayer',
                embed_dims=embed_dims,
                num_points=num_points,
                num_points_bev=num_points_bev,
                num_levels=num_levels,
                img_depth_num=img_depth_num,
                bev_depth_num=bev_depth_num,
                pc_range=point_cloud_range,
                num_groups=4)),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range))))

# dataset settings
dataset_root = '/data/dataset/RacFormer/nuscenes/'
ann_file_train = dataset_root + 'nuscenes_infos_temporal_train_newpcd.pkl'
ann_file_val = dataset_root + 'nuscenes_infos_temporal_val_newpcd.pkl'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_radar', to_float32=True),
    dict(type='LoadMultiViewRadarPointsFromFiles_v2', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='RadarPointToMultiViewDepth', grid_config=grid_config, downsample=16),
    dict(type='PointToMultiViewDepth', grid_config=grid_config, downsample=16),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'gt_depth', 'gt_bboxes_3d', 'gt_labels_3d', 'radar_points', 'radar_depth', 'radar_rcs'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_radar', to_float32=True),
    dict(type='LoadMultiViewRadarPointsFromFiles_v2', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='RadarPointToMultiViewDepth', grid_config=grid_config, downsample=16),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'radar_points', 'radar_depth', 'radar_rcs'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            data_root=dataset_root,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR',
            num_frames_per_sample=num_frames)),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_root=dataset_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        num_frames_per_sample=num_frames),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_root=dataset_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        num_frames_per_sample=num_frames))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = None
resume_from = None
workflow = [('train', 1)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)

# evaluation
eval_config = dict(interval=2)