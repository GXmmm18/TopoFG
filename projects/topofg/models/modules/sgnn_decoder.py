import copy
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import Linear, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout 
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, FEEDFORWARD_NETWORK,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.models.utils.transformer import inverse_sigmoid

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TopoFGSGNNDecoder(TransformerLayerSequence):

    def __init__(self, pc_range,*args, return_intermediate=False, **kwargs):
        super(TopoFGSGNNDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.num_pt_per_vec = 11
        self.fp16_enabled = False
            
    def forward(self,
                query,
                *args,
                reference_points=None,
                lclc_branches=None,
                lcte_branches=None,
                reg_branches = None,
                spatial_shapes = None,
                level_start_index= None,
                pts_num = 3,
                key_padding_mask=None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):
        bs = query.shape[1]
        num_query = kwargs['dn_metas']['num_query']
        dn_pad_size = kwargs['dn_metas']['dn_pad_size']
        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_lclc_rel = []
        intermediate_lclc_rel_dn = []
        intermediate_lcte_rel = []
        outputs_coords = []
        outputs_coords_dn = []

        
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                spatial_shapes = spatial_shapes,
                level_start_index= level_start_index,
                self_attn_mask=kwargs['dn_metas']['self_attn_mask'],
                **kwargs)
            

            output = output.permute(1, 0, 2)

            tmp = reg_branches[lid](output)
            tmp = tmp.view(bs, num_query, -1, pts_num)

            reference = inverse_sigmoid(reference_points).view(bs, num_query, -1, pts_num)
            tmp = tmp + reference
            
            tmp = tmp.sigmoid()
            coord = tmp.clone()
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if pts_num == 3:
                coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2] 
            outputs_coord = coord.view(bs, num_query, -1).contiguous()

            end_embeding = torch.cat([output.view(bs, num_query, self.num_pt_per_vec, -1)[:, :, -1], tmp[:,:,-1]], -1)
            start_embeding = torch.cat([output.view(bs, num_query, self.num_pt_per_vec, -1)[:, :, 0], tmp[:,:,0]], -1)
            lclc_rel_out, lclc_rel_out_dn = lclc_branches[lid](end_embeding, start_embeding, dn_pad_size)

            lcte_rel_out = lcte_branches[lid](output.view(bs, num_query, self.num_pt_per_vec, -1)[:, dn_pad_size:], \
                                              coord[:, dn_pad_size:].detach(), te_feats[lid], te_cls_scores[lid])
            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_lclc_rel.append(lclc_rel_out)
                intermediate_lclc_rel_dn.append(lclc_rel_out_dn)
                intermediate_lcte_rel.append(lcte_rel_out)
                outputs_coords_dn.append(outputs_coord[:, :dn_pad_size, :])
                outputs_coords.append(outputs_coord[:, dn_pad_size:, :])
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_lclc_rel), torch.stack(
                intermediate_lcte_rel) , torch.stack(outputs_coords_dn), torch.stack(outputs_coords), torch.stack(intermediate_lclc_rel_dn)

        return output, reference_points, lclc_rel_out, lcte_rel_out, outputs_coords_dn, outputs_coords


@TRANSFORMER_LAYER.register_module()
class SGNNDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(SGNNDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs)
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
    
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                spatial_shapes = None,
                level_start_index= None,
                self_attn_mask = None,
                **kwargs):
        num_query = kwargs['dn_metas']['num_query']
        dn_pad_size = kwargs['dn_metas']['dn_pad_size']
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            if layer == 'self_attn':
                if attn_index == 0:
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_query, -1, n_batch, n_dim).flatten(1, 2)
                    query_pos = query_pos.view(num_query, -1, n_batch, n_dim).flatten(1, 2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=self_attn_mask,
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    query = query.view(num_query, -1, n_batch, n_dim).flatten(0, 1)
                    query_pos = query_pos.view(num_query, -1, n_batch, n_dim).flatten(0, 1)
                    attn_index += 1
                    identity = query
                else:
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_query, -1, n_batch, n_dim).permute(1, 0, 2, 3).contiguous().flatten(1, 2)
                    query_pos = query_pos.view(num_query, -1, n_batch, n_dim).permute(1, 0, 2, 3).contiguous().flatten(1, 2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    query = query.view(-1, num_query, n_batch, n_dim).permute(1, 0, 2, 3).contiguous().flatten(0, 1)
                    query_pos = query_pos.view(-1, num_query, n_batch, n_dim).permute(1, 0, 2, 3).contiguous().flatten(0, 1)
                    attn_index += 1
                    identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes = spatial_shapes,
                    level_start_index= level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
