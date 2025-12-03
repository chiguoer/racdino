import queue
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion
from mmdet3d.models import builder
from torch.nn import functional as F
from torch import nn as nn
from mmcv.cnn import ConvModule
from mmdet3d.ops import Voxelization

# 导入dinov2适配器模块
from .backbones import DinoAdapter

@DETECTORS.register_module()
class RaCFormer(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 stop_prev_grad=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 num_lss_fpn=2,
                 dep_downsample=16,
                 img_lss_neck=None,
                 img_lss_view_transformer=None,
                 pre_process=None,
                 pts_neck=None,
                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 # 添加dinov2适配器参数
                 dinov2_adapter=None):
        super(RaCFormer, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                        pts_middle_encoder, pts_fusion_layer,
                                        img_backbone, pts_backbone, img_neck, pts_neck,
                                        pts_bbox_head, img_roi_head, img_rpn_head,
                                        train_cfg, test_cfg, pretrained)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True

        self.memory = {}
        self.memory_bev = {}
        self.memory_radar_bev = {}
        self.memory_dep = {}
        self.queue = queue.Queue()
        self.dep_downsample = dep_downsample

        # 初始化dinov2适配器模块
        if dinov2_adapter is not None:
            # 避免修改传入的字典，先拷贝一份
            dinov2_adapter_args = dinov2_adapter.copy()
            # 如果包含 'type' 键，则移除，因为 DinoAdapter.__init__ 不接受该参数
            if 'type' in dinov2_adapter_args:
                dinov2_adapter_args.pop('type')
            
            self.dinov2_adapter = DinoAdapter(**dinov2_adapter_args)
            
            # ResNet50的输出通道数：[256, 512, 1024, 2048] (对应4个stage)
            # 如果img_backbone没有embed_dim字段，则从depth推断标准通道数
            if 'embed_dim' in img_backbone:
                # 如果指定了embed_dim，所有stage使用相同的通道数
                backbone_channels = [img_backbone['embed_dim']] * 4
            else:
                # ResNet标准输出通道数（根据depth）
                resnet_channels = {
                    50: [256, 512, 1024, 2048],
                    101: [256, 512, 1024, 2048],
                    18: [64, 128, 256, 512],
                    34: [64, 128, 256, 512]
                }
                depth = img_backbone.get('depth', 50)
                backbone_channels = resnet_channels.get(depth, [256, 512, 1024, 2048])
            
            dinov2_embed_dim = dinov2_adapter.get('embed_dim', 768)
            
            # 语义特征融合卷积层，将dinov2输出特征与原图特征融合
            # 注意：DinoAdapter返回4个特征图，每个都是[embed_dim, H, W]
            # ResNet也输出4个特征图，通道数为backbone_channels
            self.semantic_fusion = nn.ModuleList([
                ConvModule(
                    in_channels=backbone_channels[i] + dinov2_embed_dim,
                    out_channels=backbone_channels[i],
                    kernel_size=1,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN2d'),
                    bias='auto'
                ) for i in range(4)  # 对应4个特征层级
            ])
        else:
            self.dinov2_adapter = None
            self.semantic_fusion = None

        if img_lss_neck is not None:
            self.num_lss_fpn = num_lss_fpn
            if self.num_lss_fpn == self.img_neck.num_outs:
                self.img_lss_neck = self.img_neck
            else:
                self.img_lss_neck = builder.build_neck(img_lss_neck)

        self.img_lss_view_transformer = builder.build_neck(img_lss_view_transformer)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)

        self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)

        rad_conv_layers = []
        for i in range(3):
            in_channel = radar_middle_encoder.in_channels
            if i < 2:
                out_channel = radar_middle_encoder.in_channels
            else:
                out_channel = self.pts_bbox_head.embed_dims
            rad_conv_layers.append(
                ConvModule(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN2d'),
                    bias='auto')
            )

        self.radar_bev_conv = nn.Sequential(*rad_conv_layers)

    @property
    def with_img_lss_neck(self):
        return hasattr(self, 'img_lss_neck') and self.img_lss_neck is not None

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)
        BNT, C, imH, imW = img.shape

        # 提取原始图像特征
        img_feats = self.img_backbone(img)

        # 如果有dinov2适配器，提取语义特征并融合
        if self.dinov2_adapter is not None:
            # 获取dinov2语义特征
            # DinoAdapter返回: ([f1, f2, f3, f4], x_out)
            # 其中f1-f4是4个多尺度特征图，每个形状为 [BNT, embed_dim, H, W]
            semantic_feats, _ = self.dinov2_adapter(img)

            # 融合原始特征和语义特征
            # ResNet输出4个特征图: [256, 512, 1024, 2048] channels
            # DINOv2输出4个特征图: [embed_dim] channels (384或768)
            fused_feats = []
            for i in range(min(len(img_feats), len(semantic_feats))):
                # img_feats[i]: [BNT, C_i, H_i, W_i] 其中C_i是ResNet的通道数
                # semantic_feats[i]: [BNT, embed_dim, H_dino, W_dino]
                
                # 调整DINOv2特征的空间尺寸以匹配ResNet特征
                semantic_feat_resized = F.interpolate(
                    semantic_feats[i],
                    size=img_feats[i].shape[2:],  # 匹配H和W
                    mode='bilinear',
                    align_corners=False
                )
                
                # 拼接特征: [BNT, C_i + embed_dim, H_i, W_i]
                combined = torch.cat([img_feats[i], semantic_feat_resized], dim=1)
                
                # 通过融合卷积: [BNT, C_i, H_i, W_i]
                fused = self.semantic_fusion[i](combined)
                fused_feats.append(fused)
            
            img_feats = fused_feats

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats_fpn = self.img_neck(img_feats)

        if self.with_img_lss_neck:
            img_lss_feats = self.img_lss_neck(img_feats[-self.num_lss_fpn:])
            if type(img_lss_feats) in [list, tuple]:
                img_lss_feats = img_lss_feats[0]
            _, output_dim, ouput_H, output_W = img_lss_feats.shape
            img_lss_feats = img_lss_feats.view(BNT, output_dim, ouput_H, output_W)

        return img_feats_fpn, img_lss_feats

    @auto_fp16(apply_to=('radar_points'), out_fp32=True)
    def extract_pts_feat(self, radar_points=None):
        if not self.with_pts_bbox:
            return None

        for i, radar_point in enumerate(radar_points):
            radar_point[:, 2] = 0
            radar_points[i] = radar_point

        voxels, num_points, coors = self.radar_voxelize(radar_points)
        radar_features = self.radar_voxel_encoder(voxels, num_points, coors).to(torch.float32)

        batch_size = coors[-1, 0] + 1

        radar_features = radar_features.squeeze()
        rad_bev_feas = self.radar_middle_encoder(radar_features, coors, batch_size)

        rad_bev_feas = self.radar_bev_conv(rad_bev_feas)
        return rad_bev_feas

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)

        return voxels, num_points, coors_batch

    def extract_feat(self, img, radar_points, radar_depth, radar_rcs, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, NT, C, H, W = img.size()
        N = 6
        T = NT // N
        img = img.view(B * NT, C, H, W)
        img = img.float()

        radar_depth = radar_depth.view(B * NT, 1, H, W)
        radar_depth = radar_depth.float()

        radar_rcs = radar_rcs.view(B * NT, 1, H, W)
        radar_rcs = radar_rcs.view(B * NT, 1, H, W)

        # 数据增强
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(NT)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(NT)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])
                radar_depth = pad_multiple(radar_depth, img_metas, size_divisor=img_pad_cfg['size_divisor'])
                radar_rcs = pad_multiple(radar_rcs, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        radar_depth = radar_depth.view(B, N, T, H, W)
        radar_rcs = radar_rcs.view(B, N, T, H, W)

        input_shape = img.shape[-2:]
        # 更新每张图像的实际输入形状
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            img = img.reshape(B, -1, N, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            img_feats_fpn, img_lss_feats = self.extract_img_feat(img_grad.reshape(-1, C, H, W))
            mlp_input = self.img_lss_view_transformer.get_mlp_input(img_metas)
            mlp_input_nograd = mlp_input[:, self.stop_prev_grad * N:]

            _, C_lss, h, w = img_lss_feats.shape

            img_lss_feats = img_lss_feats.view(B, img_grad.shape[1], N, C_lss, h, w)

            all_bev_feats = []
            all_depths = []

            for k in range(img_grad.shape[1]):
                img_meta_b = []
                for b in range(B):
                    img_meta = dict()
                    img_meta['lidar2img'] = img_metas[b]['lidar2img'][k * 6:(k + 1) * 6]
                    img_meta['img_shape'] = img_metas[b]['img_shape'][k * 6:(k + 1) * 6]
                    img_meta_b.append(img_meta)
                img_lss_feats_k = img_lss_feats[:, k].view(B, N, C_lss, h, w)
                bev_feat, depth = self.img_lss_view_transformer(img_lss_feats_k, img_meta_b,
                                                                mlp_input[:, k * 6:(k + 1) * 6])
                if self.pre_process:
                    bev_feat = self.pre_process_net(bev_feat)[0]
                all_bev_feats.append(bev_feat)
                all_depths.append(depth)

            with torch.no_grad():
                self.eval()
                img_feats_fpn_nograd, img_lss_feats_nograd = self.extract_img_feat(img_nograd.reshape(-1, C, H, W))
                img_lss_feats_nograd = img_lss_feats_nograd.view(B, img_nograd.shape[1], N, C_lss, h, w)

                for k in range(img_nograd.shape[1]):
                    img_meta_b = []
                    for b in range(B):
                        img_meta = dict()
                        img_meta['lidar2img'] = img_metas[b]['lidar2img'][
                                                (img_grad.shape[1] + k) * 6:(img_grad.shape[1] + k + 1) * 6]
                        img_meta['img_shape'] = img_metas[b]['img_shape'][
                                                (img_grad.shape[1] + k) * 6:(img_grad.shape[1] + k + 1) * 6]
                        img_meta_b.append(img_meta)
                    img_lss_feats_nograd_k = img_lss_feats_nograd[:, k].view(B, N, C_lss, h, w)
                    bev_feat_nograd, depth_nograd = self.img_lss_view_transformer(img_lss_feats_nograd_k, img_meta_b,
                                                                                  mlp_input_nograd[:,
                                                                                  k * 6:(k + 1) * 6])
                    if self.pre_process:
                        bev_feat_nograd = self.pre_process_net(bev_feat_nograd)[0]
                    all_bev_feats.append(bev_feat_nograd)
                    all_depths.append(depth_nograd)
                self.train()
            all_bev_feats = torch.stack(all_bev_feats, dim=1)

            img_feats = []
            for lvl in range(len(img_feats_fpn)):
                C, H, W = img_feats_fpn[lvl].shape[-3:]
                img_feat_lvl = img_feats_fpn[lvl].reshape(B, -1, 6, C, H, W)
                img_feat_nograd_lvl = img_feats_fpn_nograd[lvl].reshape(B, -1, 6, C, H, W)

                img_feat = torch.cat([img_feat_lvl, img_feat_nograd_lvl], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)

            pts_feats = self.extract_pts_feat(radar_points=radar_points)
        else:
            img_feats, img_lss_feats = self.extract_img_feat(img)
            _, C_lss, h, w = img_lss_feats.shape
            img_lss_feats = img_lss_feats.view(B, NT, C_lss, h, w)

            mlp_input = self.img_lss_view_transformer.get_mlp_input(img_metas)

            radar_bev_feats = []
            all_bev_feats = []
            all_depths = []
            for i in range(T):
                img_meta_b = []
                for b in range(B):
                    img_meta = dict()
                    img_meta['lidar2img'] = img_metas[b]['lidar2img'][i * 6:(i + 1) * 6]
                    img_meta['img_shape'] = img_metas[b]['img_shape'][i * 6:(i + 1) * 6]
                    img_meta_b.append(img_meta)
                if self.training:
                    if i == 0:
                        pts_feats = self.extract_pts_feat(radar_points=radar_points[i])

                        bev_feat, depth = self.img_lss_view_transformer(img_lss_feats[:, i * 6:(i + 1) * 6],
                                                                        radar_depth[:, :, i], radar_rcs[:, :, i],
                                                                        img_meta_b, mlp_input[:, i * 6:(i + 1) * 6])
                        if self.pre_process:
                            bev_feat = self.pre_process_net(bev_feat)[0]
                    else:
                        with torch.no_grad():
                            self.eval()
                            pts_feats = self.extract_pts_feat(radar_points=radar_points[i])

                            bev_feat, depth = self.img_lss_view_transformer(img_lss_feats[:, i * 6:(i + 1) * 6],
                                                                            radar_depth[:, :, i], radar_rcs[:, :, i],
                                                                            img_meta_b, mlp_input[:, i * 6:(i + 1) * 6])
                            if self.pre_process:
                                bev_feat = self.pre_process_net(bev_feat)[0]
                            self.train()
                else:
                    pts_feats = self.extract_pts_feat(radar_points=radar_points[i])
                    bev_feat, depth = self.img_lss_view_transformer(img_lss_feats[:, i * 6:(i + 1) * 6],
                                                                    radar_depth[:, :, i], radar_rcs[:, :, i],
                                                                    img_meta_b, mlp_input[:, i * 6:(i + 1) * 6])
                    if self.pre_process:
                        bev_feat = self.pre_process_net(bev_feat)[0]

                all_bev_feats.append(bev_feat)
                all_depths.append(depth)
                radar_bev_feats.append(pts_feats)
            all_bev_feats = torch.stack(all_bev_feats, dim=1)
            radar_bev_feats = torch.stack(radar_bev_feats, dim=1)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped, all_bev_feats, radar_bev_feats, all_depths[0]

    def forward_pts_train(self,
                          pts_feats,
                          bev_feats,
                          radar_bev_feats,
                          depth,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_depth,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats, bev_feats, radar_bev_feats, img_metas)

        loss_depth = self.img_lss_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses_pts = self.pts_bbox_head.loss(*loss_inputs)
        losses.update(losses_pts)
        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_depth=None,
                      img=None,
                      radar_points=None,
                      radar_depth=None,
                      radar_rcs=None,
                      gt_bboxes_ignore=None,
                      ):
        img_feats, bev_feats, radar_bev_feats, depth = self.extract_feat(img, radar_points, radar_depth, radar_rcs,
                                                                         img_metas)

        for i in range(len(img_metas)):
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

        losses = self.forward_pts_train(img_feats, bev_feats, radar_bev_feats, depth, gt_bboxes_3d, gt_labels_3d,
                                        gt_depth, img_metas, gt_bboxes_ignore)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, bev_feats, radar_bev_feats, img_metas, rescale=False):
        outs = self.pts_bbox_head(x, bev_feats, radar_bev_feats, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas[0], rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results

    def simple_test(self, img_metas, img=None, rescale=False, radar_points=None, radar_depth=None, radar_rcs=None,
                    **kwargs):
        world_size = get_dist_info()[1]
        if world_size == 1:  # online
            return self.simple_test_online(img_metas, img, rescale, radar_points[0], radar_depth[0], radar_rcs[0])
        else:  # offline
            return self.simple_test_offline(img_metas, img, rescale, radar_points[0], radar_depth[0], radar_rcs[0])

    def simple_test_offline(self, img_metas, img=None, rescale=False, radar_points=None, radar_depth=None,
                            radar_rcs=None):
        img_feats, bev_feats, radar_bev_feats, _ = self.extract_feat(img=img, radar_points=radar_points,
                                                                     radar_depth=radar_depth, radar_rcs=radar_rcs,
                                                                     img_metas=img_metas)

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, bev_feats, radar_bev_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def simple_test_online(self, img_metas, img=None, rescale=False, radar_points=None, radar_depth=None,
                           radar_rcs=None):
        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape
        img = img.reshape(B, N // 6, 6, C, H, W)
        radar_depth = radar_depth.reshape(B, N // 6, 6, 1, H, W)
        radar_rcs = radar_rcs.reshape(B, N // 6, 6, 1, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // 6

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_large, bev_feats_large, radar_bev_feats_large, img_metas_large, dep_large = [], [], [], [], []

        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if isinstance(img_metas[0][k], list):
                    img_metas_curr[0][k] = [img_metas[0][k][m] for m in img_indices]

            if img_filenames[img_indices[0]] in self.memory:
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
                bev_feat_curr = self.memory_bev[img_filenames[img_indices[0]]]
                radar_bev_feat_curr = self.memory_radar_bev[img_filenames[img_indices[0]]]

            else:
                img_curr_large = img[:, i]  # [B, 6, C, H, W]
                radar_dep_curr_large = radar_depth[:, i]  # [B, 6, C, H, W]
                radar_rcs_curr_large = radar_rcs[:, i]  # [B, 6, C, H, W]
                radar_points_curr = [radar_points[i]]
                img_feats_curr, bev_feat_curr, radar_bev_feat_curr, _ = self.extract_feat(img_curr_large,
                                                                                          radar_points_curr,
                                                                                          radar_dep_curr_large,
                                                                                          radar_rcs_curr_large,
                                                                                          img_metas_curr)

                self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                self.memory_bev[img_filenames[img_indices[0]]] = bev_feat_curr
                self.memory_radar_bev[img_filenames[img_indices[0]]] = radar_bev_feat_curr

                self.queue.put(img_filenames[img_indices[0]])

            img_feats_large.append(img_feats_curr)
            img_metas_large.append(img_metas_curr)
            bev_feats_large.append(bev_feat_curr)
            radar_bev_feats_large.append(radar_bev_feat_curr)

        feat_levels = len(img_feats_large[0])
        img_feats_large_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_large[i][j] for i in range(len(img_feats_large))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_large_reorganized.append(feat_l)

        img_metas_large_reorganized = img_metas_large[0]
        for i in range(1, len(img_metas_large)):
            for k, v in img_metas_large[i][0].items():
                if isinstance(v, list):
                    img_metas_large_reorganized[0][k].extend(v)

        img_feats = img_feats_large_reorganized
        img_metas = img_metas_large_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        lss_bev_feats = torch.cat(bev_feats_large, dim=1)
        radar_bev_feat = torch.cat(radar_bev_feats_large, dim=1)

        bbox_list = [dict() for _ in range(1)]
        bbox_pts = self.simple_test_pts(img_feats, lss_bev_feats, radar_bev_feat, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        while self.queue.qsize() >= 16:
            pop_key = self.queue.get()
            self.memory.pop(pop_key)
            self.memory_bev.pop(pop_key)
            self.memory_radar_bev.pop(pop_key)

        return bbox_list