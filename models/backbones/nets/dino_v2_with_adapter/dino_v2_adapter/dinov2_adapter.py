# adaptation of https://github.com/czczup/ViT-Adapter

import sys
import os
# 动态获取当前文件所在目录，避免硬编码路径
current_dir = os.path.dirname(os.path.abspath(__file__))
nets_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
if nets_dir not in sys.path:
    sys.path.append(nets_dir)

import logging
import math
import timeit
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from ..dino_v2.layers import MemEffAttention
from ..dino_v2.layers import NestedTensorBlock as Block
from ..dino_v2.model.vision_transformer import (
    DinoVisionTransformer,
)
from ...ops.modules import MSDeformAttn

from .adapter_modules import (
    InteractionBlock,
    InteractionBlockWithCls,
    SpatialPriorModule,
    deform_inputs,
)

_logger = logging.getLogger(__name__)

from mmdet3d.models.builder import BACKBONES

@BACKBONES.register_module()
class DinoAdapter(DinoVisionTransformer):
    # patch_size in dino_v2: 14 instead of 16
    def __init__(self, num_heads=12, pretrain_size=518, pretrained_vit=True, patch_size=14, embed_dim=768, depth=12,
                 mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention), conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
                 with_cffn=True, cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=False, pretrained=None,
                 use_extra_extractor=True, with_cp=False, freeze_dino=True, *args, **kwargs):

        super().__init__(img_size=pretrain_size, num_heads=num_heads, patch_size=14, embed_dim=embed_dim, depth=depth, mlp_ratio=mlp_ratio,
                         block_fn=block_fn, freeze=True, *args, **kwargs)

        self.pretrain_size = pretrain_size

        if pretrained_vit:
            # 优先从本地weight和pretrain文件夹加载权重
            if num_heads == 6:
                model_name = 'dinov2_vits14'
                weight_filename = 'dinov2_vits14_pretrain.pth'
            else:
                model_name = 'dinov2_vitb14'
                weight_filename = 'dinov2_vitb14_pretrain.pth'
            
            # 尝试从多个路径加载权重
            weight_paths = [
                os.path.join('weight', weight_filename),  # 优先从weight文件夹
                os.path.join('pretrain', weight_filename),  # 其次从pretrain文件夹
                os.path.join(os.path.expanduser("~"), ".cache", "dinov2", weight_filename),  # 本地缓存
            ]
            
            state_dict = None
            loaded_from = None
            
            # 尝试从本地路径加载
            for weight_path in weight_paths:
                if os.path.exists(weight_path):
                    try:
                        state_dict = torch.load(weight_path, map_location="cpu")
                        loaded_from = weight_path
                        _logger.info(f"成功从 {weight_path} 加载DINOv2预训练权重")
                        break
                    except Exception as e:
                        _logger.warning(f"从 {weight_path} 加载权重失败: {e}")
                        continue
            
            # 如果本地加载失败，尝试从torch.hub加载
            if state_dict is None:
                try:
                    _logger.info(f"尝试从torch.hub加载 {model_name}...")
                    pretrained_model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
                    state_dict = pretrained_model.state_dict()
                    loaded_from = "torch.hub"
                    _logger.info(f"成功从torch.hub加载DINOv2预训练权重")
                except Exception as e:
                    _logger.warning(f"从torch.hub加载失败: {e}")
            
            # 加载权重到模型
            if state_dict is not None:
                try:
                    self.load_state_dict(state_dict, strict=True)
                    _logger.info(f"DINOv2权重加载成功 (来源: {loaded_from})")
                except Exception as e:
                    _logger.warning(f"加载state_dict时出错: {e}，尝试非严格模式...")
                    try:
                        self.load_state_dict(state_dict, strict=False)
                        _logger.info(f"DINOv2权重加载成功 (非严格模式)")
                    except Exception as e2:
                        _logger.error(f"无法加载DINOv2预训练权重: {e2}。将使用随机初始化。")
            else:
                _logger.warning("无法找到DINOv2预训练权重，将使用随机初始化。")

        if freeze_dino:
            for param in self.parameters():
                param.requires_grad = False

        # self.num_classes = 80
        self.mask_token = None
        self.num_block = len(self.blocks)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        # embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)

        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                                    init_values=init_values, drop_path=self.drop_path_rate,
                                    norm_layer=self.norm_layer, with_cffn=with_cffn,
                                    cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                    extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                                    with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02) uses timm --> not wanted
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(1, self.pretrain_size // 14, self.pretrain_size // 14, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        # 添加统一的卷积输出尺寸计算函数
        def conv_output_size(input_size, kernel_size=3, stride=2, padding=1):
            return (input_size + 2*padding - kernel_size) // stride + 1

        # 1. 预处理 Padding (必须在最开始进行)
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        h_padded, w_padded = x.shape[2], x.shape[3]

        # 计算 stem 输出尺寸 (kernel=7, stride=4, padding=3)
        H1_spm = conv_output_size(h_padded, kernel_size=7, stride=4, padding=3)
        W1_spm = conv_output_size(w_padded, kernel_size=7, stride=4, padding=3)

        # 2. SPM forward
        c1, c2, c3, c4 = self.spm(x)
        
        # 验证 stem 输出尺寸
        _, _, actual_H1, actual_W1 = c1.shape
        assert actual_H1 == H1_spm, f"c1 height mismatch: {actual_H1} != {H1_spm}"
        assert actual_W1 == W1_spm, f"c1 width mismatch: {actual_W1} != {W1_spm}"

        # 从 SPM 输出推导实际空间形状
        bs_spm, L2, dim_spm = c2.shape
        _, L3, _ = c3.shape
        _, L4, _ = c4.shape
        
        # 使用统一公式计算后续层尺寸 (kernel=3, stride=2, padding=1)
        H2_spm = conv_output_size(H1_spm)
        W2_spm = conv_output_size(W1_spm)
        H3_spm = conv_output_size(H2_spm)
        W3_spm = conv_output_size(W2_spm)
        H4_spm = conv_output_size(H3_spm)
        W4_spm = conv_output_size(W3_spm)
        
        # 验证尺寸
        assert L2 == H2_spm * W2_spm, f"L2 mismatch: {L2} != {H2_spm}×{W2_spm}"
        assert L3 == H3_spm * W3_spm, f"L3 mismatch: {L3} != {H3_spm}×{W3_spm}"
        assert L4 == H4_spm * W4_spm, f"L4 mismatch: {L4} != {H4_spm}×{W4_spm}"

        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        
        # 3. 使用实际 SPM 形状构建 deform_inputs
        deform_inputs1, deform_inputs2 = deform_inputs(x, c2_shape=(H2_spm, W2_spm), 
                                                        c3_shape=(H3_spm, W3_spm), 
                                                        c4_shape=(H4_spm, W4_spm))
        
        # 保存 SPM 各层的精确形状，用于 DWConv
        spm_shapes = [(H2_spm, W2_spm), (H3_spm, W3_spm), (H4_spm, W4_spm)]

        # Patch Embedding forward
        x_patched = self.patch_embed(x)
        W_vit = w_padded // self.patch_size
        H_vit = h_padded // self.patch_size
        W_adapt = w_padded // 16
        H_adapt = h_padded // 16

        bs, n, dim = x_patched.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_vit, W_vit)
        x = x_patched + pos_embed
        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, 0]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H_adapt, W_adapt, spm_shapes=spm_shapes)
            outs.append(x.transpose(1, 2).view(bs, dim, H_vit, W_vit).contiguous())

        # Split & Reshape
        # 从SPM输出获取各层的大小
        c2_size = H2_spm * W2_spm  # 使用精确的 SPM 形状
        c3_size = H3_spm * W3_spm
        c4_size = H4_spm * W4_spm
        
        # 按照原始大小分割c
        c2 = c[:, 0:c2_size, :]
        c3 = c[:, c2_size:c2_size + c3_size, :]
        c4 = c[:, c2_size + c3_size:c2_size + c3_size + c4_size, :]
        
        # 使用精确的 SPM 形状进行重塑
        c2 = c2.transpose(1, 2).view(bs, dim, H2_spm, W2_spm).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H3_spm, W3_spm).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H4_spm, W4_spm).contiguous()

        # 确保上采样后的 c2 与 c1 形状匹配
        up_c2 = self.up(c2)
        if up_c2.size(2) != c1.size(2) or up_c2.size(3) != c1.size(3):
            up_c2 = F.interpolate(
                up_c2, 
                size=(c1.size(2), c1.size(3)), 
                mode='bilinear', 
                align_corners=False
            )
        c1 = up_c2 + c1

        x_out = x.transpose(1, 2).view(bs, dim, H_vit, W_vit).contiguous()

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        
        # 3. 输出裁剪
        # 需要同时裁剪 f1-f4 (来自SPM) 和 x_out (来自ViT)
        # 注意：h, w 是原始输入尺寸（padding 之前）
        if pad_h > 0 or pad_w > 0:
            # 裁剪 x_out (基于 patch_size)
            target_h = h // self.patch_size
            target_w = w // self.patch_size
            x_out = x_out[:, :, :target_h, :target_w]
            
            # 裁剪 f1-f4 (基于 SPM 下采样率)
            # c1: stride 4
            # c2: stride 8
            # c3: stride 16
            # c4: stride 32
            f1 = f1[:, :, :h//4, :w//4]
            f2 = f2[:, :, :h//8, :w//8]
            f3 = f3[:, :, :h//16, :w//16]
            f4 = f4[:, :, :h//32, :w//32]
        
        return [f1, f2, f3, f4], x_out
    
    def extract_intermediate_features(self, x):
        # 添加统一的卷积输出尺寸计算函数
        def conv_output_size(input_size, kernel_size=3, stride=2, padding=1):
            return (input_size + 2*padding - kernel_size) // stride + 1

        # 同样应用 padding 逻辑
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
        h_padded, w_padded = x.shape[2], x.shape[3]
        
        # 计算 stem 输出尺寸 (kernel=7, stride=4, padding=3)
        H1_spm = conv_output_size(h_padded, kernel_size=7, stride=4, padding=3)
        W1_spm = conv_output_size(w_padded, kernel_size=7, stride=4, padding=3)
        
        # SPM forward (先执行以获取实际特征形状)
        c1, c2, c3, c4 = self.spm(x)
        
        # 从 SPM 输出推导实际空间形状
        _, _, actual_H1, actual_W1 = c1.shape
        assert actual_H1 == H1_spm, f"c1 height mismatch: {actual_H1} != {H1_spm}"
        assert actual_W1 == W1_spm, f"c1 width mismatch: {actual_W1} != {W1_spm}"
        
        bs_spm, L2, dim_spm = c2.shape
        _, L3, _ = c3.shape
        _, L4, _ = c4.shape
        
        # 使用统一公式计算后续层尺寸
        H2_spm = conv_output_size(H1_spm)
        W2_spm = conv_output_size(W1_spm)
        H3_spm = conv_output_size(H2_spm)
        W3_spm = conv_output_size(W2_spm)
        H4_spm = conv_output_size(H3_spm)
        W4_spm = conv_output_size(W3_spm)
        
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        
        deform_inputs1, deform_inputs2 = deform_inputs(x, c2_shape=(H2_spm, W2_spm), 
                                                        c3_shape=(H3_spm, W3_spm), 
                                                        c4_shape=(H4_spm, W4_spm))
        
        # 保存 SPM 各层的精确形状，用于 DWConv
        spm_shapes = [(H2_spm, W2_spm), (H3_spm, W3_spm), (H4_spm, W4_spm)]

        # Patch Embedding forward
        x_patched = self.patch_embed(x)
        W_vit = w_padded // self.patch_size
        H_vit = h_padded // self.patch_size
        W_adapt = w_padded // 16
        H_adapt = h_padded // 16

        bs, n, dim = x_patched.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_vit, W_vit)
        x = x_patched + pos_embed
        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, 0]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H_adapt, W_adapt, spm_shapes=spm_shapes)
            outs.append(x.transpose(1, 2).view(bs, dim, H_vit, W_vit).contiguous())

        return outs, x, c, cls
