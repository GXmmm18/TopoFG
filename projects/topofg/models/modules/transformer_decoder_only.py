import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER

from projects.bevformer.modules.decoder import CustomMSDeformableAttention
from projects.bevformer.modules.spatial_cross_attention import \
    MSDeformableAttention3D
from projects.bevformer.modules.temporal_self_attention import \
    TemporalSelfAttention


@TRANSFORMER.register_module()
class TopoFGTransformerDecoderOnly(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 decoder=None,
                 embed_dims=256,
                 pts_dim=3,
                 **kwargs):
        super(TopoFGTransformerDecoderOnly, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.fp16_enabled = False
        self.pts_dim = pts_dim
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        pass

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                bev_feats_ms,
                query_feat,
                query_pos,
                reference_points,
                bev_h,
                bev_w,
                lclc_branches=None,
                lcte_branches=None,
                reg_branches=None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):

        reference_points = reference_points.flatten(1, 2).sigmoid()
    
        init_reference_out = reference_points

        query = query_feat.permute(1, 0, 2)
        query_pos = query_pos.flatten(1, 2).permute(1, 0, 2)
        bev_embed_ms_flatten = []
        spatial_flatten = []
        for lvl in range(len(bev_feats_ms)):
            B, C, H, W = bev_feats_ms[lvl].shape
            bev_embed_ms_flatten.append(bev_feats_ms[lvl].permute(0, 2, 3, 1).reshape(B, -1, C).permute(1, 0, 2))
            spatial_flatten.append((H, W))
        bev_embed_ms_flatten = torch.cat(bev_embed_ms_flatten, dim=0)
        spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=query_feat.device)
        level_start_index = torch.cat((spatial_flatten.new_zeros((1,)), spatial_flatten.prod(1).cumsum(0)[:-1]))
        inter_states, inter_lclc_rel, inter_lcte_rel, outputs_coords_dn, outputs_coords, inter_lclc_rel_dn = self.decoder(
            query=query,
            key=None,
            value=bev_embed_ms_flatten,
            query_pos=query_pos,
            reference_points=reference_points,
            lclc_branches=lclc_branches,
            lcte_branches=lcte_branches,
            reg_branches = reg_branches,
            spatial_shapes=spatial_flatten,
            level_start_index=level_start_index,
            pts_num = self.pts_dim,
            te_feats=te_feats,
            te_cls_scores=te_cls_scores,
            bev_feats=bev_feats_ms[1].flatten(-2),
            **kwargs)



        return inter_states, inter_lclc_rel, inter_lcte_rel, outputs_coords_dn, outputs_coords, inter_lclc_rel_dn
