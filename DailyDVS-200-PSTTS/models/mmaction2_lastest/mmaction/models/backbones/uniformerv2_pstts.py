# ==============================================================================
# PSTTS: A Plug-and-Play Token Selector for Efficient Event-based Spatio-temporal Representation Modeling
# Copyright (c) The PSTTS Authors.
# Licensed under The MIT License.
# Written by Nan Yang.
# Modified from mmaction.
# ==============================================================================

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union
import math
import torch
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import _load_checkpoint
from torch import nn
import torch.nn.functional as F
from mmaction.registry import MODELS

import random
import os
import cv2
import numpy as np

logger = MMLogger.get_current_instance()

MODEL_PATH = 'https://download.openmmlab.com/mmaction/v1.0/recognition'
_MODELS = {
    'ViT-B/16':
    os.path.join(MODEL_PATH, 'uniformerv2/clipVisualEncoder',
                 'vit-base-p16-res224_clip-rgb_20221219-b8a5da86.pth'),
    'ViT-L/14':
    os.path.join(MODEL_PATH, 'uniformerv2/clipVisualEncoder',
                 'vit-large-p14-res224_clip-rgb_20221219-9de7543e.pth'),
    'ViT-L/14_336':
    os.path.join(MODEL_PATH, 'uniformerv2/clipVisualEncoder',
                 'vit-large-p14-res336_clip-rgb_20221219-d370f9e5.pth'),
}


class QuickGELU(BaseModule):
    """Quick GELU function. Forked from https://github.com/openai/CLIP/blob/d50
    d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py.

    Args:
        x (torch.Tensor): The input features of shape :math:`(B, N, C)`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Local_MHRA(BaseModule):
    """Local MHRA.

    Args:
        d_model (int): Number of input channels.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        pos_kernel_size (int): Kernel size of local MHRA.
            Defaults to 3.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        dw_reduction: float = 1.5,
        pos_kernel_size: int = 3,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(
                re_d_model,
                re_d_model,
                kernel_size=(pos_kernel_size, 1, 1),
                stride=(1, 1, 1),
                padding=(padding, 0, 0),
                groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        logger.info('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_embed(x)


class ResidualAttentionBlock(BaseModule):
    """Local UniBlock.

    Args:
        d_model (int): Number of input channels.
        n_head (int): Number of attention head.
        drop_path (float): Stochastic depth rate.
            Defaults to 0.0.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA.
            Defaults to True.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        drop_path: float = 0.0,
        dw_reduction: float = 1.5,
        no_lmhra: bool = False,
        double_lmhra: bool = True,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.n_head = n_head
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        logger.info(f'Drop path rate: {drop_path}')

        self.no_lmhra = no_lmhra
        self.double_lmhra = double_lmhra
        logger.info(f'No L_MHRA: {no_lmhra}')
        logger.info(f'Double L_MHRA: {double_lmhra}')
        if not no_lmhra:
            self.lmhra1 = Local_MHRA(d_model, dw_reduction=dw_reduction)
            if double_lmhra:
                self.lmhra2 = Local_MHRA(d_model, dw_reduction=dw_reduction)

        # spatial
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x: torch.Tensor, T: int = 8, index_token: torch.Tensor = None, index_token2: torch.Tensor = None) -> torch.Tensor:
        # x: 1+HW, NT, C
        if not self.no_lmhra:
            # Local MHRA
            tmp_x = x[1:, :, :]
            L, NT, C = tmp_x.shape
            N = NT // T
            H = W = int(L**0.5)
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # MHSA
        XX = x.clone()
        xx = x.gather(dim=0, index=index_token)
        XXX = xx.clone()
        xx = xx.gather(dim=0, index=index_token2)
        xx = xx + self.drop_path(self.attention(self.ln_1(xx)))
        # Local MHRA
        if not self.no_lmhra and self.double_lmhra:
            tmp_x = x[1:, :, :]
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # FFN
        xx = xx + self.drop_path(self.mlp(self.ln_2(xx)))

        x = XXX.to(x.dtype).scatter_(0, index_token2, xx.to(x.dtype))
        x = XX.to(x.dtype).scatter_(0, index_token, x.to(x.dtype))
        return x


class Extractor(BaseModule):
    """Global UniBlock.

    Args:
        d_model (int): Number of input channels.
        n_head (int): Number of attention head.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers.
            Defaults to 4.0.
        drop_out (float): Stochastic dropout rate.
            Defaults to 0.0.
        drop_path (float): Stochastic depth rate.
            Defaults to 0.0.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_factor: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        logger.info(f'Drop path rate: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_mlp)),
                         ('gelu', QuickGELU()),
                         ('dropout', nn.Dropout(dropout)),
                         ('c_proj', nn.Linear(d_mlp, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)

        # zero init
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def attention(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T
             ) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T
             ) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T
             ) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

def compute_tmr(windows_tensor):
    B, T, num_windows, window_size, _ = windows_tensor.shape
    
    windows_next = windows_tensor[:, :-1]  # (B, T-1, 196, 16, 16)
    windows_current = windows_tensor[:, 1:]  # (B, T-1, 196, 16, 16)
    
    # Compute the mean
    mu_current = windows_current.mean(dim=(3, 4), keepdim=True)  # (B, T-1, 196, 1, 1)
    mu_next = windows_next.mean(dim=(3, 4), keepdim=True)  # (B, T-1, 196, 1, 1)
    
    # Compute variance and covariance
    sigma_current = (windows_current - mu_current).pow(2).mean(dim=(3, 4), keepdim=True)  # (B, T-1, 196, 1, 1)
    sigma_next = (windows_next - mu_next).pow(2).mean(dim=(3, 4), keepdim=True)  # (B, T-1, 196, 1, 1)
    sigma_cross = ((windows_current - mu_current) * (windows_next - mu_next)).mean(dim=(3, 4), keepdim=True)  # (B, T-1, 196, 1, 1)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算 TMR
    tmr_numerator = (2 * mu_current * mu_next + C1) * (2 * sigma_cross + C2/2)
    tmr_denominator = (mu_current.pow(2) + mu_next.pow(2) + C1) * (torch.sqrt(sigma_current) * torch.sqrt(sigma_next) + C2/2)
    tmr_matrix = tmr_numerator / tmr_denominator  # (B, T-1, 196, 1, 1)
    
    # 调整形状为 (B, T-1, 196, 196)
    tmr_matrix = tmr_matrix.squeeze(-1).squeeze(-1)  # (B, T-1, 196)
    tmr_matrix = tmr_matrix.unsqueeze(3).expand(-1, -1, -1, num_windows)  # (B, T-1, 196, 196)
    
    return tmr_matrix

class Transformer(BaseModule):
    """Backbone:

    Args:
        width (int): Number of input channels in local UniBlock.
        layers (int): Number of layers of local UniBlock.
        heads (int): Number of attention head in local UniBlock.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA
            in local UniBlock. Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
        dw_reduction: float = 1.5,
        no_lmhra: bool = True,
        double_lmhra: bool = False,
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,
        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.T = t_size
        self.return_list = return_list
        # backbone
        b_dpr = [
            x.item()
            for x in torch.linspace(0, backbone_drop_path_rate, layers)
        ]
        self.resblocks = ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                drop_path=b_dpr[i],
                dw_reduction=dw_reduction,
                no_lmhra=no_lmhra,
                double_lmhra=double_lmhra,
            ) for i in range(layers)
        ])

        # global block
        assert n_layers == len(return_list)
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))
        self.dpe = ModuleList([
            nn.Conv3d(
                n_dim,
                n_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=n_dim) for _ in range(n_layers)
        ])
        for m in self.dpe:
            nn.init.constant_(m.bias, 0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.dec = ModuleList([
            Extractor(
                n_dim,
                n_head,
                mlp_factor=mlp_factor,
                dropout=mlp_dropout[i],
                drop_path=dpr[i],
            ) for i in range(n_layers)
        ])
        # weight sum
        self.norm = nn.LayerNorm(n_dim)
        self.balance = nn.Parameter(torch.zeros((n_dim)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, tcm: torch.Tensor, tcm_window: torch.Tensor) -> torch.Tensor:
        # NT L C tcm_window
        C0 = tcm_window.shape[-1]
        T_down = self.T
        L, NT, C = x.shape
        N = NT // T_down
        H = W = int((L - 1)**0.5)
        cls_token = self.temporal_cls_token.repeat(1, N, 1)

        tcm = tcm.squeeze(2)
        hw = tcm.shape[0]

        # Spatial Token Purification
        factor = 1.75
        cc = torch.sum(tcm, dim=0) / (hw * factor)
        cc = cc.repeat(hw, 1)
        gts = tcm >= cc
        K = torch.sum(gts, dim=0)
        class_pad = torch.ones(1, NT).to(tcm.dtype).cuda() * torch.max(tcm)
        scores_final = torch.cat([class_pad, tcm], dim=0)
        index_token = torch.topk(scores_final, k=K.max()+1, dim=0, largest=True, sorted=False)[1]

        # ​​Phase 1 Retention​
        tcm2 = tcm.gather(dim=0, index=index_token[1:]-1)
        num0 = index_token.shape[0]-1

        index_token = index_token.unsqueeze(2)
        tcm_window = tcm_window.permute(1, 0, 2)
        tcm_window = tcm_window.gather(dim=0, index=index_token[1:].repeat(1, 1, C0)-1)
        tcm_window2 = tcm_window.contiguous().view(num0, N, T_down, 16, 16)
        tcm_window2 = tcm_window2.permute(1, 2, 0, 3, 4)

        index_token = index_token.repeat(1, 1, x.shape[-1])
        

        # Temporal Token Selection
        # for T0
        s_agg_final = torch.zeros(N, 1, num0,1).to(tcm.dtype).cuda() 
        # 计算temporal motion redundancy score
        tmr = compute_tmr(tcm_window2)
        s_agg = tmr.sum(dim=-1, keepdim=True)#.softmax(dim=2)
        s_agg = s_agg / s_agg.sum(dim=2, keepdim=True)
        s_agg_final = torch.cat([s_agg_final, s_agg], dim=1)
        s_agg_finals = s_agg_final[...,0].permute(2, 0, 1).reshape(num0, N * (T_down))
        
        j = -1
        index_token2 = None
        for i, resblock in enumerate(self.resblocks):
            
            if i == 0 or i == 3:
                x_m = x.gather(dim=0, index=index_token[1:])
                scores = torch.norm(x_m, dim=[2], p=2)

                scores_final2 = tcm2 * (1. - 1 * s_agg_finals) * scores 

                # ​​Phase 2 Retention​
                factor = 4.5
                cc2 = torch.sum(scores_final2, dim=0) / (num0 * factor)
                cc2 = cc2.repeat(num0, 1)
                gts2 = scores_final2 > cc2
                K2 = torch.sum(gts2, dim=0)
                # for class_token
                class_pad2 = torch.ones(1, NT).to(scores_final2.dtype).cuda() * torch.max(scores_final2)
                scores_final2 = torch.cat([class_pad2, scores_final2], dim=0)
            
                index_token2 = torch.topk(scores_final2, k=K2.max()+1, dim=0, largest=True, sorted=False)[1]
                index_token2 = index_token2.unsqueeze(2)
                index_token2 = index_token2.repeat(1, 1, x.shape[-1])

            
            x = resblock(x, T_down, index_token, index_token2)
            if i in self.return_list:
                j += 1
                tmp_x = x.clone()
                tmp_x = tmp_x.view(L, N, T_down, C)
                # dpe
                _, tmp_feats = tmp_x[:1], tmp_x[1:]
                tmp_feats = tmp_feats.permute(1, 3, 2,
                                              0).reshape(N, C, T_down, H, W)
                tmp_feats = self.dpe[j](tmp_feats.clone()).view(
                    N, C, T_down, L - 1).permute(3, 0, 2, 1).contiguous()
                tmp_x[1:] = tmp_x[1:] + tmp_feats
                # global block
                tmp_x = tmp_x.flatten(1, 2)
                assert index_token.shape[1] == tmp_x.shape[1]
                assert index_token.shape[2] == tmp_x.shape[2]
                tmp_x = tmp_x.gather(dim=0, index=index_token)
                tmp_x = tmp_x.gather(dim=0, index=index_token2)
                tmp_x = tmp_x.view(index_token2.shape[0], N, T_down, C)

                tmp_x = tmp_x.permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C
                cls_token = self.dec[j](cls_token, tmp_x)

        weight = self.sigmoid(self.balance)
        residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C
        out = self.norm((1 - weight) * cls_token[0, :, :] + weight * residual)
        return out

class EventSpatialTemporalFilter(nn.Module):
    def __init__(self,
                 spatial_radius: int = 3,
                 temporal_sigma: float = 0.1,
                 spatial_sigma: float = 1.5,
                 use_polarity: bool = True):
        """
        Args:
            spatial_radius: Neighborhood radius (actual window size is 2*radius+1)
            temporal_sigma: Similarity Gaussian kernel parameter for temporal continuity score
            spatial_sigma: Gaussian kernel parameter for spatial distance
            use_polarity: Whether to use polarity consistency constraint
        """
        super().__init__()
        self.radius = spatial_radius
        self.temp_sigma = temporal_sigma
        self.spat_sigma = spatial_sigma
        self.use_polarity = use_polarity
        
        # Precompute spatial coordinate offsets (for the local neighborhood of the convolution)
        y, x = torch.meshgrid(
            torch.arange(-spatial_radius, spatial_radius+1),
            torch.arange(-spatial_radius, spatial_radius+1)
        )
        self.register_buffer('spatial_offsets', torch.stack([x, y], dim=-1).float())  # [H,W,2]
        
    def forward(self, 
                event_map: torch.Tensor,          # [B,1,H,W] Event Tensor
                time_continuity: torch.Tensor,     # [B,1,H,W] Temporal Continuity Score Map
                polarity: Optional[torch.Tensor] = None  # [B,1,H,W] ​​Polarity (+1/-1), optional​
               ) -> torch.Tensor:
        """
        Return the enhanced event map after spatio-temporal aggregation [B,1,H,W]
        """
        B, C, H, W = event_map.shape
        device = event_map.device
        
        # 1. Construct Spatio-Temporal Weight Matrix​
        # ​​Spatial Distance Weight​​ [H,W,1]
        spatial_dist = torch.norm(self.spatial_offsets, dim=-1, keepdim=True)  # [H,W,1]
        spatial_weights = torch.exp(-0.5 * (spatial_dist / self.spat_sigma)**2)
        
        # 2. Extract Neighborhood Features
        # Unfold Local Patches​​ [B,CK,K,H,W], where ​​K = number of neighborhood pixels​*​
        K = (2*self.radius + 1)**2
        padded_time = F.pad(time_continuity, [self.radius]*4, mode='reflect')
        time_patches = F.unfold(padded_time, kernel_size=2*self.radius+1)  # [B,K,L], L=H*W
        time_patches = time_patches.view(B, 1, K, H, W)  # [B,1,K,H,W]
        
        # 3. Calculate Temporal Continuity Similarity Weight​
        center_time = time_continuity.unsqueeze(2)  # [B,1,1,H,W]
        time_diff = (time_patches - center_time).abs()  # [B,1,K,H,W]
        time_weights = torch.exp(-0.5 * (time_diff / self.temp_sigma)**2)  # [B,1,K,H,W]
        
        # 4. Optional Polarity Consistency Constraint​
        if self.use_polarity and polarity is not None:
            padded_pol = F.pad(polarity, [self.radius]*4, mode='constant', value=0)
            pol_patches = F.unfold(padded_pol, kernel_size=2*self.radius+1)  # [B,K,L]
            pol_patches = pol_patches.view(B, 1, K, H, W)  # [B,1,K,H,W]
            polarity_mask = (pol_patches * polarity.unsqueeze(2)) > 0  # 同极性为True
            time_weights = time_weights * polarity_mask.float()
        
        # 5. Combine Spatio-Temporal Weights and Normalize
        combined_weights = spatial_weights.view(1,1,K,1,1) * time_weights  # [B,1,K,H,W]
        norm_weights = combined_weights / (combined_weights.sum(dim=2, keepdim=True) + 1e-6)
        
        # 6. Apply Weighted Aggregation​
        padded_events = F.pad(event_map, [self.radius]*4, mode='constant')
        event_patches = F.unfold(padded_events, kernel_size=2*self.radius+1)  # [B,K,L]
        event_patches = event_patches.view(B, 1, K, H, W)  # [B,1,K,H,W]
        
        enhanced_events = (event_patches * norm_weights).sum(dim=2)  # [B,1,H,W]
        
        return enhanced_events

@MODELS.register_module()
class UniFormerV2(BaseModule):
    """UniFormerV2:

    A pytorch implement of: `UniFormerV2: Spatiotemporal
    Learning by Arming Image ViTs with Video UniFormer
    <https://arxiv.org/abs/2211.09552>`

    Args:
        input_resolution (int): Number of input resolution.
            Defaults to 224.
        patch_size (int): Number of patch size.
            Defaults to 16.
        width (int): Number of input channels in local UniBlock.
            Defaults to 768.
        layers (int): Number of layers of local UniBlock.
            Defaults to 12.
        heads (int): Number of attention head in local UniBlock.
            Defaults to 12.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        temporal_downsample (bool): Whether downsampling temporal dimentison.
            Defaults to False.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA in local UniBlock.
            Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        clip_pretrained (bool): Whether to load pretrained CLIP visual encoder.
            Defaults to True.
        pretrained (str): Name of pretrained model.
            Defaults to None.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        # backbone
        input_resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
        kernel_size: int = 3,
        dw_reduction: float = 1.5,
        temporal_downsample: bool = False,
        no_lmhra: bool = True,
        double_lmhra: bool = False,
        # global block
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,
        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        # pretrain
        clip_pretrained: bool = True,
        pretrained: Optional[str] = None,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.clip_pretrained = clip_pretrained
        self.input_resolution = input_resolution
        padding = (kernel_size - 1) // 2
        if temporal_downsample:
            self.conv1 = nn.Conv3d(
                3,
                width, (kernel_size, patch_size, patch_size),
                (2, patch_size, patch_size), (padding, 0, 0),
                bias=False)
            t_size = t_size // 2
            self.time_embed = nn.AvgPool3d(
                kernel_size=(kernel_size, patch_size, patch_size),
                stride=(2, patch_size, patch_size), 
                padding=(padding, 0, 0))
        else:
            self.conv1 = nn.Conv3d(
                3,
                width, (1, patch_size, patch_size),
                (1, patch_size, patch_size), (0, 0, 0),
                bias=False)
            self.time_embed = nn.AvgPool3d(
                kernel_size=(1, patch_size, patch_size),
                stride=(1, patch_size, patch_size), 
                padding=(0, 0, 0))

        self.stfilter = EventSpatialTemporalFilter(
                                    spatial_radius=2,
                                    temporal_sigma=0.2,
                                    spatial_sigma=1.7,
                                    use_polarity=False
                                ).cuda()

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(
            width,
            layers,
            heads,
            dw_reduction=dw_reduction,
            backbone_drop_path_rate=backbone_drop_path_rate,
            t_size=t_size,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            return_list=return_list,
            n_layers=n_layers,
            n_dim=n_dim,
            n_head=n_head,
            mlp_factor=mlp_factor,
            drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout,
        )

        

    def _inflate_weight(self,
                        weight_2d: torch.Tensor,
                        time_dim: int,
                        center: bool = True) -> torch.Tensor:
        logger.info(f'Init center: {center}')
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def _load_pretrained(self, pretrained: str = None) -> None:
        """Load CLIP pretrained visual encoder.

        The visual encoder is extracted from CLIP.
        https://github.com/openai/CLIP

        Args:
            pretrained (str): Model name of pretrained CLIP visual encoder.
                Defaults to None.
        """
        assert pretrained is not None, \
            'please specify clip pretraied checkpoint'

        model_path = _MODELS[pretrained]
        logger.info(f'Load CLIP pretrained model from {model_path}')
        state_dict = _load_checkpoint(model_path, map_location='cpu')
        state_dict_3d = self.state_dict()
        for k in state_dict.keys():
            if k in state_dict_3d.keys(
            ) and state_dict[k].shape != state_dict_3d[k].shape:
                if len(state_dict_3d[k].shape) <= 2:
                    logger.info(f'Ignore: {k}')
                    continue
                logger.info(f'Inflate: {k}, {state_dict[k].shape}' +
                            f' => {state_dict_3d[k].shape}')
                time_dim = state_dict_3d[k].shape[2]
                state_dict[k] = self._inflate_weight(state_dict[k], time_dim)
        self.load_state_dict(state_dict, strict=False)

    def init_weights(self):
        """Initialize the weights in backbone."""
        if self.clip_pretrained:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            self._load_pretrained(self.pretrained)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:

        x = input_tensor[:, :3, :, :, :]
        # Temporal Continuity Score Map​
        tcm = input_tensor[:, 3:, :, :, :]

        N0, C0, T0, H0, W0 = tcm.shape

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        N, C, T, H, W = x.shape
        
        # window_size = patch_size
        window_size = 16
        # Partition Windows for Temporal Token Selection​
        tcm_window = tcm.permute(0, 2, 1, 3, 4).reshape(N * T, 1, H0, W0)
        # Spatial Continuity​
        tcm_window = self.stfilter(tcm_window, tcm_window)
        tcm_window = tcm_window.squeeze(1).unfold(1, window_size, window_size).unfold(2, window_size, window_size)
        tcm_window = tcm_window.contiguous().view(N * T, -1, window_size * window_size)

        # N, C, T, H, W
        tcm = self.time_embed(tcm)
        tcm = tcm.permute(0, 2, 1, 3, 4).reshape(N * T, 1, H, W)

        # Spatial Continuity​
        tcm = self.stfilter(tcm, tcm)
        # N * T, C, H, W
        tcm = tcm.reshape(N * T, 1, H * W).permute(2, 0, 1)
        
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)

        # N * T, H * W + 1, C
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(x, tcm, tcm_window)
        return out
