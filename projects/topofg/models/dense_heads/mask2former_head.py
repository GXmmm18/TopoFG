import cv2
import torch
import numpy as np
import torch.nn as nn
from shapely import affinity
from mmdet.models import HEADS
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from shapely.geometry import LineString
from mmdet.datasets.pipelines import to_tensor
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils.positional_encoding import SinePositionalEncoding

from ...core.lane.util import normalize_3dlane, denormalize_3dlane

# @TRANSFORMER_LAYER_SEQUENCE.register_module()
@HEADS.register_module()
class MaskEncoder(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        num_transformer_feat_level=3,
        num_query=200,
        segm_decoder=None,
        num_classes=1,
        dn_enabled=False,
        dn_group_num=5,
        dn_label_noise_ratio=0.2,
        pts2mask_noise_scale=0.2,
        bev_h=100,
        bev_w=200,
        mask_noise_scale = 0.1,
        **kwargs,

    ):


        super(MaskEncoder, self).__init__(**kwargs)
        self.embed_dims = embed_dims
        self.patch_size = [50, 100]
        self.num_query = num_query
        self.num_classes = num_classes
        self.num_transformer_feat_level = num_transformer_feat_level
        self.instance_query_feat = nn.Embedding(self.num_query, self.embed_dims)

    #     # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, self.embed_dims)
        # self.cls_embed = nn.Linear(self.embed_dims, self.num_classes+1)
        self.mask_embed = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.decoder_positional_encoding = SinePositionalEncoding(num_feats=self.embed_dims // 2, normalize=True)

        self.num_heads = segm_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_segm_decoder_layers = segm_decoder.num_layers
        self.segm_decoder = build_transformer_layer_sequence(segm_decoder)
        self.dn_enabled = dn_enabled
        self.dn_group_num = dn_group_num
        self.pts2mask_noise_scale = pts2mask_noise_scale
        if dn_enabled:
            self.label_enc = nn.Embedding(num_classes, self.embed_dims)
        self.mask_noise_scale = mask_noise_scale
        self.lane_noise_scale = 0.5
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = bev_h, bev_w
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2

    def forward(self, bev_embed_ms, targets=None, pc_range=None, **kwargs):

        bs=bev_embed_ms[0].shape[0]
        mask_feat = bev_embed_ms[0].to(torch.float32)

        ms_memorys = bev_embed_ms[:0:-1]

        decoder_inputs = []
        decoder_pos_encodings = []
        size_list = []
        for i in range(self.num_transformer_feat_level):
            size_list.append(ms_memorys[i].shape[-2:])
            decoder_input = ms_memorys[i]
            decoder_input = decoder_input.flatten(2)
            decoder_input = decoder_input.permute(0, 2, 1)  # [bs, hw, c]
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros((bs,) + ms_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_pos_encoding = self.decoder_positional_encoding(mask).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_pos_encodings.append(decoder_pos_encoding)

        instance_query_feat = self.instance_query_feat.weight[0:self.num_query].unsqueeze(0).expand(bs, -1, -1)
        instance_query_pos = None
        if self.dn_enabled and self.training:
            (instance_query_feat, know_lane_embed, padding_mask_3level, self_attn_mask, mask_dict, known_bid, map_known_indice, dn_single_pad, dn_pad_size, known_masks) = (
                self.prepare_for_dn_input_pts2mask(mask_feat, instance_query_feat, size_list, targets, pc_range)
            )
        else:
            mask_dict, known_masks, known_bid, map_known_indice, know_lane_embed = (None, None, None, None, None)
            dn_single_pad = dn_pad_size = 0

        dn_metas = dict(
            mask_dict=mask_dict,
            known_masks=known_masks,
            known_bid=known_bid,
            map_known_indice=map_known_indice,
            know_lane_embed=know_lane_embed,
            dn_enabled=self.dn_enabled,
            dn_pad_size=dn_pad_size,
            dn_single_pad=dn_single_pad,
        )
        mask_pred_list = []
        mask_pred, attn_mask = self._forward_head(instance_query_feat, mask_feat, ms_memorys[0].shape[-2:])
        mask_pred_list.append(mask_pred)
        if self.dn_enabled == False or self.training == False:
            self_attn_mask = torch.zeros([self.num_query, self.num_query]).bool().to(bev_embed_ms[0].device)
            self_attn_mask[self.num_query :, 0 : self.num_query] = True
            self_attn_mask[0 : self.num_query, self.num_query :] = True
            self_attn_mask = self_attn_mask.unsqueeze(0).expand(bs, -1, -1)
            self_attn_mask = self_attn_mask.unsqueeze(1)
            self_attn_mask = self_attn_mask.repeat((1, self.num_heads, 1, 1))
            self_attn_mask = self_attn_mask.flatten(0, 1)
            dn_metas["num_query"] = self.num_query
            dn_metas["self_attn_mask"] = None


        else:
            attn_mask = attn_mask.view([bs, self.num_heads, -1, attn_mask.shape[-1]])
            attn_mask[:, :, :-self.num_query] = padding_mask_3level[0]
            attn_mask = attn_mask.flatten(0, 1)
            self_attn_mask = self_attn_mask.clone()
            self_attn_mask[: dn_pad_size + self.num_query, dn_pad_size + self.num_query :] = True
            self_attn_mask[dn_pad_size + self.num_query :, : dn_pad_size + self.num_query] = True
            dn_metas["num_query"] = self.num_query + dn_pad_size
            dn_metas["self_attn_mask"] = self_attn_mask.clone()

        for i in range(self.num_segm_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.segm_decoder.layers[i]
            instance_query_feat = layer(
                query=instance_query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=instance_query_pos,
                key_pos=decoder_pos_encodings[level_idx],
                cross_attn_mask=attn_mask,
                self_attn_mask=self_attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
            mask_pred, attn_mask = self._forward_head(
                instance_query_feat, mask_feat, ms_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:]
            )

            if self.dn_enabled and i != self.num_segm_decoder_layers - 1 and self.training:
                padding_mask = padding_mask_3level[(i + 1) % self.num_segm_decoder_layers]
                attn_mask = attn_mask.view([bs, self.num_heads, -1, attn_mask.shape[-1]])
                attn_mask[:, :, :-self.num_query] = padding_mask
                attn_mask = attn_mask.flatten(0, 1)
            mask_pred_list.append(mask_pred)

        return mask_pred_list, instance_query_feat, dn_metas

    def prepare_for_dn_input_pts2mask(self, mask_feat, instance_query_feat, size_lists, targets, pc_range):
        bs, f_dim, h, w = mask_feat.shape
        device = instance_query_feat.device
        num_query = instance_query_feat.shape[1]
        pt_dim = 3 if targets[0]['gt_lanes'].shape[-1] % 3 ==0 else 2
        num_per_vec = targets[0]['gt_lanes'].shape[-1] // pt_dim
        gt_lanes_3d = [gt['gt_lanes'].view(-1, num_per_vec, pt_dim) for gt in targets]
        targets = [
            {
                "gt_pts_list": gt_lanes_3d[idx][..., :2],
                "gt_masks_list": gt['gt_masks_list'],
                "labels": gt['gt_labels'].long(),
                "gt_bboxes_list": self.bbox_condi(gt_lanes_3d[idx][..., :2]).cuda(),
            }
            for idx, gt in enumerate(targets)
        ]
        known = [torch.ones_like(t["labels"], device=device) for t in targets]
        known_num = [sum(k) for k in known]
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets]).clone()
        bboxes = torch.cat([t["gt_bboxes_list"] for t in targets]).clone()
        masks = [
            F.interpolate(targets[i]["gt_masks_list"].unsqueeze(1).float(), size=mask_feat.shape[-2:], mode="bilinear") for i in range(len(targets))
        ]

        gt_pts_list = torch.cat([t["gt_pts_list"] for t in targets]).clone()
        lanes = torch.cat([t for t in gt_lanes_3d])
        dn_single_pad = int(max(known_num))
        dn_pad_size = int(dn_single_pad * self.dn_group_num)
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(self.dn_group_num).view(-1)

        known_labels = labels.repeat(self.dn_group_num).view(-1)
        known_bid = batch_idx.repeat(self.dn_group_num).view(-1)
        known_bboxes = bboxes.repeat(self.dn_group_num, 1)
        known_masks = torch.cat(masks, dim=0).squeeze(1).repeat(self.dn_group_num, 1, 1)
        known_gt_pts_list = gt_pts_list.repeat(self.dn_group_num, 1, 1)
        known_lanes = lanes.repeat(self.dn_group_num, 1, 1)
        known_lane_expand = known_lanes.clone()

        yx_scale = known_gt_pts_list[..., 1].max() / known_gt_pts_list[..., 0].max()
        pts_distance = torch.sqrt(((known_gt_pts_list[:, 0] - known_gt_pts_list[:, 1]) ** 2).sum(-1))
        rand_prob = torch.rand(known_gt_pts_list.shape).cuda()
        diff = (rand_prob * pts_distance[:, None, None]) * self.pts2mask_noise_scale * torch.tensor((1, yx_scale)).cuda()[None]
        noise_known_gt_pts_list = known_gt_pts_list + diff
        known_lane_expand[..., :2] = known_lane_expand[..., :2] + diff
        segm_list = []
        scale_y = self.canvas_size[0] / self.patch_size[0]
        scale_x = self.canvas_size[1] / self.patch_size[1]
        trans_y = self.canvas_size[0] / 2
        trans_x = self.canvas_size[1] / 2
        for pts in noise_known_gt_pts_list:
            instance_segm = np.zeros(self.canvas_size, dtype=np.uint8)
            line_ego = affinity.scale(LineString(pts.cpu()), scale_x, scale_y, origin=(0, 0))
            line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
            coords = coords.reshape((-1, 2))
            assert len(coords) >= 2
            coords[:, 1] = instance_segm.shape[0] - coords[:, 1]  # 翻转y轴 cv2中y轴向下，GT中y轴向上
            cv2.polylines(instance_segm, np.int32([coords]), False, color=1, thickness=2)
            segm_list.append(instance_segm)

        segm_tensor = to_tensor(segm_list).cuda()
        masks_for_attn = (F.interpolate(segm_tensor.float().unsqueeze(1), size=size_lists[-1], mode="nearest") <= 1e-8).squeeze(1)
        padding_mask_3level = []
        for i in range(len(size_lists)):
            padding_mask = torch.ones([bs, dn_pad_size, size_lists[i][0] * size_lists[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        masks_3level = []
        known_boxes_expand = torch.stack(
            (
                (known_bboxes[..., 3] + known_bboxes[..., 1]) / 2,
                (known_bboxes[..., 2] + known_bboxes[..., 0]) / 2,
                (known_bboxes[..., 3] - known_bboxes[..., 1]),
                (known_bboxes[..., 2] - known_bboxes[..., 0]),
            ),
            dim=1,
        )
        diff = torch.zeros_like(known_boxes_expand)
        diff[..., :2] = torch.abs(known_boxes_expand[..., :2]) / 4 * self.mask_noise_scale
        diff[..., 2:] = known_boxes_expand[..., 2:] / 2 * self.mask_noise_scale
        delta_masks = torch.mul((torch.rand_like(known_boxes_expand) * 2 - 1.0), diff)
        is_scale = torch.rand_like(known_boxes_expand[..., 0])
        new_masks = []
        scale_noise = torch.rand_like(known_boxes_expand[..., 0]).to(known_boxes_expand) * self.mask_noise_scale * 1.5

        scale_size = (torch.tensor(size_lists[-1]).float().to(known_boxes_expand)[None] * (1 + scale_noise)[:, None]).long() + 1
        delta_center = (torch.tensor(size_lists[-1])[None].to(known_boxes_expand) - scale_size).to(known_boxes_expand) * (
            known_boxes_expand[..., :2] / torch.tensor(size_lists[-1]).to(known_boxes_expand)[None]
        )
        scale_size = scale_size.tolist()

        for mask, delta_mask, sc, noise_scale, dc in zip(masks_for_attn, delta_masks, is_scale, scale_size, delta_center):
            mask_scale = F.interpolate(mask[None][None].float(), noise_scale, mode="nearest")[0][0]
            x_, y_ = torch.where(mask_scale < 0.5)
            x_ += dc[0].long()
            y_ += dc[1].long()
            delta_x = delta_mask[0]
            delta_y = delta_mask[1]
            x_ = x_ + delta_x
            y_ = y_ + delta_y
            x_ = x_.clamp(min=0, max=size_lists[-1][-2] - 1)
            y_ = y_.clamp(min=0, max=size_lists[-1][-1] - 1)
            mask = torch.ones_like(mask, dtype=torch.bool)
            mask[x_.long(), y_.long()] = False
            new_masks.append(mask)

        new_masks = torch.stack(new_masks)
        noise_mask = new_masks.flatten(1)
        masks_3level.append(noise_mask)
        noise_mask = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_lists[-2], mode="nearest") > 0.5).flatten(1)
        masks_3level.append(noise_mask)
        noise_mask = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_lists[0], mode="nearest") > 0.5).flatten(1)
        masks_3level.append(noise_mask)
     
        known_lane_expand = normalize_3dlane(known_lane_expand, pc_range)

        dn_instance_query_feat = torch.zeros([dn_pad_size, instance_query_feat.shape[-1]], device=device).expand(bs, -1, -1)
        instance_query_feat = torch.cat([dn_instance_query_feat, instance_query_feat], dim=1)
        know_lane_embed = inverse_sigmoid(known_lane_expand)
        know_lane_embed = know_lane_embed.unsqueeze(0).repeat(bs, 1, 1, 1)

        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + dn_single_pad * i for i in range(self.dn_group_num)]).long()

        if len(known_bid):
            known_labels_expand = known_labels.clone()
            known_features = self.label_enc(known_labels_expand)
            instance_query_feat[known_bid.long(), map_known_indice] = known_features

        total_size = dn_pad_size + num_query
        attn_mask = torch.ones([total_size, total_size], device=device) < 0

        # original
        # match query cannot see the reconstruct
        attn_mask[dn_pad_size:, :dn_pad_size] = True
        attn_mask[:dn_pad_size, dn_pad_size:] = True
        for i in range(self.dn_group_num):
            if i == 0:
                attn_mask[dn_single_pad * i : dn_single_pad * (i + 1), dn_single_pad * (i + 1) : dn_pad_size] = True
            if i == self.dn_group_num - 1:
                attn_mask[dn_single_pad * i : dn_single_pad * (i + 1), : dn_single_pad * i] = True
            else:
                attn_mask[dn_single_pad * i : dn_single_pad * (i + 1), dn_single_pad * (i + 1) : dn_pad_size] = True
                attn_mask[dn_single_pad * i : dn_single_pad * (i + 1), : dn_single_pad * i] = True

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indice)] = masks_3level[2 - i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, self.num_heads, 1, 1])

        mask_dict = {
            "known_indice": torch.as_tensor(known_indice).long(),
            "batch_idx": torch.as_tensor(batch_idx).long(),
            "map_known_indice": torch.as_tensor(map_known_indice, device=device).long(),
            "known_lbs_masks": (known_labels, known_masks),
            "pad_size": dn_pad_size,
            "dn_single_pad": dn_single_pad,
            "known_num": known_num,
        }
        return instance_query_feat, know_lane_embed, padding_mask_3level, attn_mask, mask_dict, known_bid, map_known_indice, dn_single_pad, dn_pad_size, known_masks

    def _forward_head(self, decoder_out, mask_feature, attn_mask_target_size):

        decoder_out = self.segm_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        # cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)
        attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode="bilinear", align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return mask_pred, attn_mask

    def bbox_condi(self, lane):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        num_gt, num_pts, _ = lane.shape
        instance_list = lane.cpu().numpy()
        instance_bbox_list = []
        for instance in instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append((instance[:, 0].min(), instance[:, 1].min(), instance[:, 0].max(), instance[:, 1].max()))

        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(dtype=torch.float32)
        instance_bbox_tensor[:, 0] = torch.clamp(instance_bbox_tensor[:, 0], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 1] = torch.clamp(instance_bbox_tensor[:, 1], min=-self.max_y, max=self.max_y)
        instance_bbox_tensor[:, 2] = torch.clamp(instance_bbox_tensor[:, 2], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 3] = torch.clamp(instance_bbox_tensor[:, 3], min=-self.max_y, max=self.max_y)
        return instance_bbox_tensor


class DetrTransformerDecoder(BaseModule):

    def __init__(self, num_layers, layer_cfg, post_norm_cfg=dict(type="LN"), return_intermediate=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.post_norm_cfg = post_norm_cfg
        self.return_intermediate = return_intermediate
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([DetrTransformerDecoderLayer(**self.layer_cfg) for _ in range(self.num_layers)])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg, self.embed_dims)[1]

    def forward(self, query, key, value, query_pos, key_pos, key_padding_mask, **kwargs):
        """Forward function of decoder
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor): The input key, has shape (bs, num_keys, dim).
            value (Tensor): The input value with the same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).

        Returns:
            Tensor: The forwarded results will have shape
            (num_decoder_layers, bs, num_queries, dim) if
            `return_intermediate` is `True` else (1, bs, num_queries, dim).
        """
        intermediate = []
        for layer in self.layers:
            query = layer(query, key=key, value=value, query_pos=query_pos, key_pos=key_pos, key_padding_mask=key_padding_mask, **kwargs)
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
        query = self.post_norm(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query.unsqueeze(0)


class DetrTransformerDecoderLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0, batch_first=True),
        cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0, batch_first=True),
        ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True)),
        norm_cfg=dict(type="LN"),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        if "batch_first" not in self.self_attn_cfg:
            self.self_attn_cfg["batch_first"] = True
        else:
            assert (
                self.self_attn_cfg["batch_first"] is True
            ), "First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag."

        if "batch_first" not in self.cross_attn_cfg:
            self.cross_attn_cfg["batch_first"] = True
        else:
            assert (
                self.cross_attn_cfg["batch_first"] is True
            ), "First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag."

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(3)]
        self.norms = ModuleList(norms_list)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        self_attn_mask=None,
        cross_attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        query = self.self_attn(query=query, key=query, value=query, query_pos=query_pos, key_pos=query_pos, attn_mask=self_attn_mask, **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MaskTransformerDecoder(DetrTransformerDecoder):
    """Decoder of Mask2Former."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([MaskTransformerDecoderLayer(**self.layer_cfg) for _ in range(self.num_layers)])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg, self.embed_dims)[1]


class MaskTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in Mask2Former transformer."""

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        self_attn_mask=None,
        cross_attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        query = self.self_attn(query=query, key=query, value=query, query_pos=query_pos, key_pos=query_pos, attn_mask=self_attn_mask, **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query
