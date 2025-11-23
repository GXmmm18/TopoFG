import copy
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.utils import build_transformer
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS, build_loss, build_head
from mmdet.models.utils.positional_encoding import SinePositionalEncoding
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean

from ...utils.position_encoding import SeqPositionalEncoding
from ...core.lane.util import sample_coords

@HEADS.register_module()
class TopoFGHead(AnchorFreeHead):

    def __init__(self,
                num_classes,
                in_channels,
                num_query=100,
                transformer=None,
                lclc_head=None,
                lcte_head=None,
                bbox_coder=None,
                num_reg_fcs=2,
                code_weights=None,
                bev_h=30,
                bev_w=30,
                num_pts_per_vec=11,
                pc_range=None,
                pts_dim =3,
                dn_weight=1.0,
                sync_cls_avg_factor=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    bg_cls_weight=0.1,
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_segm_mask=None,
                loss_segm_dice=None,
                 train_cfg=dict(
                    assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'

            self.assigner = build_assigner(assigner)

            assert 'segm_assigner' in train_cfg, 'segment assigner should be provided '\
                'when train_cfg is set.'
            segm_assigner = train_cfg['segm_assigner']

            self.assigner_segm = build_assigner(segm_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.pts_dim = pts_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dn_weight = dn_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_segm_mask = build_loss(loss_segm_mask)
        self.loss_segm_dice = build_loss(loss_segm_dice)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_pts_per_vec = num_pts_per_vec
        self.thresh_of_mask_for_pos = 0.3
        if lclc_head is not None:
            self.lclc_cfg = lclc_head

        if lcte_head is not None:
            self.lcte_cfg = lcte_head

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.patch_size = (bev_w/2, bev_h/2)
        self.fp16_enabled = False

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 6
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.gt_c_save = self.code_size
        self.pos_enc = SeqPositionalEncoding(self.embed_dims*2, max_len=11)  # 11个点的位置编码
        self.poc_embed = nn.Linear(self.embed_dims * 2, self.embed_dims*2)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_reg_fcs = num_reg_fcs
        self._init_layers()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size // self.num_pts_per_vec))
        reg_branch = nn.Sequential(*reg_branch)

        lclc_branch = build_head(self.lclc_cfg)
        lcte_branch = build_head(self.lcte_cfg)

        te_embed_branch = []
        in_channels = self.embed_dims
        for _ in range(self.num_reg_fcs - 1):
            te_embed_branch.append(nn.Sequential(
                    Linear(in_channels, 2 * self.embed_dims),
                    nn.ReLU(),
                    nn.Dropout(0.1)))
            in_channels = 2 * self.embed_dims
        te_embed_branch.append(Linear(2 * self.embed_dims, self.embed_dims))
        te_embed_branch = nn.Sequential(*te_embed_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        self.query_emb = nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims))
        num_pred = self.transformer.decoder.num_layers
        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.lclc_branches = _get_clones(lclc_branch, num_pred)
        self.lcte_branches = _get_clones(lcte_branch, num_pred)
        # self.te_embed_branches = _get_clones(te_embed_branch, num_pred)
        positional_encoding = SinePositionalEncoding(num_feats=self.embed_dims // 2, normalize=True)
        pos_embed_mask = torch.zeros(1, self.bev_h, self.bev_w).bool()
        pos_embed_map = positional_encoding(pos_embed_mask)
        self.pos_embed_map = pos_embed_map.flatten(2).permute(0, 2, 1)

        self.query_embedding = nn.Embedding(11, self.embed_dims * 2)

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, bev_feats_ms, img_metas, te_feats, te_cls_scores, **kwargs):
        
        bs = bev_feats_ms[0].shape[0]
        dtype = bev_feats_ms[0].dtype
        dn_pad_size = kwargs['dn_metas']['dn_pad_size']
        object_query_embeds, object_query_pos, reference_points = self.query_generater(**kwargs)
        num_query = kwargs['dn_metas']['num_query']
        object_query_embeds = object_query_embeds.to(dtype)


        outputs = self.transformer(
            bev_feats_ms,
            object_query_embeds,
            object_query_pos,
            reference_points,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            lclc_branches=self.lclc_branches,
            lcte_branches=self.lcte_branches,
            reg_branches = self.reg_branches,
            te_feats=te_feats,
            te_cls_scores=te_cls_scores,
            img_metas=img_metas,
            mask_prob=None,
            **kwargs
        )

        hs, lclc_rel_out, lcte_rel_out, outputs_coords_dn, outputs_coords, lclc_rel_out_dn = outputs

        hs = hs.permute(0, 2, 1, 3)

        outputs_classes = []
        outputs_classes_dn = []        
        for lvl in range(hs.shape[0]):

            outputs_class = self.cls_branches[lvl](hs[lvl]).view(bs, num_query, self.num_pts_per_vec, -1).mean(2)
            outputs_classes.append(outputs_class[:, dn_pad_size:, : ])
            outputs_classes_dn.append(outputs_class[:, :dn_pad_size, : ])
        outputs_classes = torch.stack(outputs_classes)
        outputs_classes_dn = torch.stack(outputs_classes_dn)


        outs = {
            'all_cls_scores': outputs_classes,
            'all_lanes_preds': outputs_coords,
            'all_lclc_preds': lclc_rel_out,
            'all_lcte_preds': lcte_rel_out,
            'outputs_coords_dn': outputs_coords_dn,
            'all_cls_scores_dn': outputs_classes_dn,
            "all_lclc_preds_dn": lclc_rel_out_dn,
        }

        return outs

    def query_generater(self, ins_query, dn_metas):

        dn_pad_size = dn_metas['dn_pad_size']
        mask_pred = dn_metas.pop('mask_pred')
        known_masks, known_bid, map_known_indice = dn_metas.pop('known_masks'), dn_metas.pop('known_bid'), dn_metas.pop('map_known_indice')
        object_query_embeds = self.query_embedding.weight
        query_pos, query = torch.split(
            object_query_embeds, self.embed_dims, dim=1)

        bs, nq, h, w = mask_pred.shape
        pos_embed_map = self.pos_embed_map.repeat(bs, 1, 1).to(ins_query.device)
        mask_prob = mask_pred.clone().sigmoid()

        mask_bool = mask_prob > self.thresh_of_mask_for_pos
        coords = sample_coords(mask_bool, ins_query, self.pc_range, patch_size=self.patch_size)
        ref_pts = inverse_sigmoid(coords[..., :self.pts_dim])

        mask_prob = mask_prob.flatten(2).permute(0, 2, 1)
        if known_masks is not None:
            mask_prob = mask_prob.clone()
            mask_prob = mask_prob.permute(0, 2, 1).contiguous()
            mask_prob[known_bid, map_known_indice] = known_masks.view(known_masks.shape[0], -1)
            mask_prob = mask_prob.permute(0, 2, 1).contiguous()
        mask_bool = mask_prob > self.thresh_of_mask_for_pos
        
        ins_query_emb = self.query_emb(ins_query.detach())
        seq_pos = self.pos_enc(torch.arange(11).to(ins_query.device))
        seq_pts_embeds = self.poc_embed(seq_pos)
        seq_pts_query_pos, seq_pts_query = torch.split(
            seq_pts_embeds, self.embed_dims, dim=-1)
        seq_pts_query = query + seq_pts_query
        seq_pts_query_pos = query_pos + seq_pts_query_pos
        seq_mask_aware_query = (ins_query_emb.unsqueeze(2) + seq_pts_query.unsqueeze(1)).flatten(1, 2)
        
        mask = ~mask_bool
        atten_weight = torch.ones(bs, h * w, nq, device=mask_bool.device)
        atten_weight[mask] = mask_prob[mask]
        atten_weight = atten_weight
        mask_aware_query_pos = torch.einsum('bnd,bnq->bqd', pos_embed_map, atten_weight)
        mask_aware_query_pos = mask_aware_query_pos / torch.abs(mask_aware_query_pos).max()
        mask_aware_query_pos = torch.nan_to_num(mask_aware_query_pos, 0)
        seq_mask_query_pos = (mask_aware_query_pos.unsqueeze(2) + seq_pts_query_pos.unsqueeze(1))

        return seq_mask_aware_query, seq_mask_query_pos, ref_pts
    

    def _get_target_single(self,
                           cls_score,
                           lanes_pred,
                           lclc_pred,
                           gt_labels,
                           gt_lanes,
                           gt_lane_adj,
                           gt_bboxes_ignore=None):

        num_bboxes = lanes_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(lanes_pred, cls_score, gt_lanes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, lanes_pred,
                                              gt_lanes)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lanes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_lanes.new_ones(num_bboxes)

        # bbox targets
        gt_c = gt_lanes.shape[-1]
        if gt_c == 0:
            gt_c = self.gt_c_save
            sampling_result.pos_gt_bboxes = torch.zeros((0, gt_c)).to(sampling_result.pos_gt_bboxes.device)
        else:
            self.gt_c_save = gt_c

        bbox_targets = torch.zeros_like(lanes_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(lanes_pred)
        bbox_weights[pos_inds] = 1.0
        # DETR

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        lclc_target = torch.zeros_like(lclc_pred.squeeze(-1), dtype=gt_lane_adj.dtype, device=lclc_pred.device)
        xs = pos_inds.unsqueeze(-1).repeat(1, pos_inds.size(0))
        ys = pos_inds.unsqueeze(0).repeat(pos_inds.size(0), 1)
        lclc_target[xs, ys] = gt_lane_adj[pos_assigned_gt_inds][:, pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights, lclc_target,
                pos_inds, neg_inds, pos_assigned_gt_inds)

    def _get_target_single_segm(self, mask_pred, gt_masks):
        # sample points
        num_queries = mask_pred.shape[0]
        # assign and sample
        assign_result = self.assigner_segm.assign(mask_pred, gt_masks)
        sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((num_queries,))
        mask_weights[pos_inds] = 1.0

        return mask_targets, mask_weights, pos_inds, neg_inds

    def get_targets(self,
                    cls_scores_list,
                    lanes_preds_list,
                    lclc_preds_list,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_lane_adj_list,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, lanes_targets_list, lanes_weights_list, lclc_targets_list,
            pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, lanes_preds_list, lclc_preds_list,
            gt_labels_list, gt_lanes_list, gt_lane_adj_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list, pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )
        return (labels_list, label_weights_list, lanes_targets_list, lanes_weights_list, lclc_targets_list,
                num_total_pos, num_total_neg, assign_result)

    def get_targets_segm(self, mask_preds, gt_masks):
        (   
            mask_targets_list,
            mask_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single_segm,
            mask_preds,
            gt_masks,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return mask_targets_list, mask_weights_list, num_total_pos, num_total_neg

    def pts_loss_single(self,
                    cls_scores,
                    lanes_preds,
                    lclc_preds,
                    lcte_preds,
                    te_assign_result,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_lane_adj_list,
                    gt_lane_lcte_adj_list,
                    layer_index,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]
        lclc_preds_list = [lclc_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, lanes_preds_list, lclc_preds_list, 
                                           gt_lanes_list, gt_labels_list, gt_lane_adj_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, lclc_targets_list,
         num_total_pos, num_total_neg, assign_result) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        lclc_targets = torch.cat(lclc_targets_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))
        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan, :self.code_size], 
            bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        # lclc loss
        lclc_targets = 1 - lclc_targets.view(-1).long()
        lclc_preds = lclc_preds.view(-1, 1)
        loss_lclc = self.lclc_branches[layer_index].loss_rel(lclc_preds, lclc_targets)

        loss_lcte = self.lcte_branches[layer_index].loss(lcte_preds, gt_lane_lcte_adj_list, assign_result, te_assign_result)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, loss_lclc, loss_lcte
    
    def segm_loss_single(self,     
        segm_mask_preds=None,
        gt_segm_mask=None,
        ):


        segm_mask_shape = segm_mask_preds.shape[-2:]
        segm_mask_points_num = segm_mask_shape[0] * segm_mask_shape[1]
        segm_targets = self.get_targets_segm(segm_mask_preds, gt_segm_mask)
        (segm_mask_targets_list, segm_mask_weights_list, segm_num_total_pos, segm_num_total_neg) = (segm_targets)
        
        # shape (num_total_gts, h, w)
        segm_mask_targets = torch.cat(segm_mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        segm_mask_weights = torch.stack(segm_mask_weights_list, dim=0)

        num_total_masks = reduce_mean(segm_mask_preds.new_tensor([segm_num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        segm_mask_preds = segm_mask_preds[segm_mask_weights > 0]
        if segm_mask_targets.shape[0] == 0:
            # zero match
            loss_dice = segm_mask_preds.sum()
            loss_mask = segm_mask_preds.sum()
            return loss_mask, loss_dice

        # dice loss
        loss_segm_dice = self.loss_segm_dice(segm_mask_preds, segm_mask_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        segm_mask_preds = segm_mask_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        segm_mask_targets = segm_mask_targets.reshape(-1)
        loss_segm_mask = self.loss_segm_mask(
            segm_mask_preds,
            segm_mask_targets,
            avg_factor=num_total_masks * segm_mask_points_num,
        )

        return loss_segm_dice, loss_segm_mask

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,   
             mask_dict,
             gt_masks_list,
             gt_lanes_3d,
             gt_labels_list,
             gt_lane_adj,
             gt_lane_lcte_adj,
             te_assign_results,
             gt_bboxes_ignore=None,
             img_metas=None):

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        all_cls_scores = preds_dicts['all_cls_scores']
        all_lanes_preds = preds_dicts['all_lanes_preds']
        all_lclc_preds = preds_dicts['all_lclc_preds']
        all_lcte_preds = preds_dicts['all_lcte_preds']
        all_segm_mask_pred = preds_dicts['all_segm_mask_pred']

        num_dec_layers = len(all_cls_scores)

        gt_lanes_list = [lane for lane in gt_lanes_3d]

        all_gt_lanes_list = [gt_lanes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_lane_adj_list = [gt_lane_adj for _ in range(num_dec_layers)]
        all_gt_lane_lcte_adj_list = [gt_lane_lcte_adj for _ in range(num_dec_layers)]
        all_gt_segms_mask_list = [gt_masks_list for _ in range(len(all_segm_mask_pred))]

        layer_index = [i for i in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_lclc, losses_lcte = multi_apply(
            self.pts_loss_single, all_cls_scores, all_lanes_preds, all_lclc_preds, all_lcte_preds, te_assign_results,
            all_gt_lanes_list, all_gt_labels_list, all_gt_lane_adj_list, all_gt_lane_lcte_adj_list, layer_index)
        
        losses_segm_dice, losses_segm_mask = multi_apply(
            self.segm_loss_single, all_segm_mask_pred, all_gt_segms_mask_list)

        if mask_dict is not None:
            dn_loss_segm_dice, dn_loss_segm_mask, dn_loss_cls, dn_loss_pts, dn_loss_lclc = self.calcul_dn_loss(gt_lanes_3d, mask_dict)
        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_lane_cls'] = losses_cls[-1]
        loss_dict['loss_lane_reg'] = losses_bbox[-1]
        loss_dict['loss_lclc'] = losses_lclc[-1]
        loss_dict['loss_lclc_dn'] = dn_loss_lclc[-1]
        loss_dict['loss_lcte'] = losses_lcte[-1]
        loss_dict['losses_segm_dice'] = losses_segm_dice[-1]
        loss_dict['losses_segm_mask'] = losses_segm_mask[-1]


        # loss from other decoder layers
        num_segm_layer = 0
        for losses_segm_dice_i, losses_segm_mask_i  in zip(losses_segm_dice[:-1], losses_segm_mask[:-1]):
            loss_dict[f'd{num_segm_layer}.losses_segm_dice'] = losses_segm_dice_i
            loss_dict[f'd{num_segm_layer}.losses_segm_mask'] = losses_segm_mask_i
            num_segm_layer += 1

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_lclc_i, loss_lcte_i in zip(
            losses_cls[:-1], losses_bbox[:-1], losses_lclc[:-1], losses_lcte[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_lane_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_lane_reg'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_lclc'] = loss_lclc_i
            loss_dict[f'd{num_dec_layer}.loss_lcte'] = loss_lcte_i
            num_dec_layer += 1
        
        # dn_loss
        if mask_dict is not None:
            loss_dict["loss_segm_dice_dn"] = dn_loss_segm_dice[-1]
            loss_dict["loss_segm_mask_dn"] = dn_loss_segm_mask[-1]          
            
            loss_dict["loss_cls_dn"] = dn_loss_cls[-1]
            loss_dict["loss_pts_dn"] = dn_loss_pts[-1]

            num_dec_layer = 0
            for loss_cls_i, loss_pts_i, loss_lclc_i in zip(dn_loss_cls[:-1], dn_loss_pts[:-1], dn_loss_lclc[:-1]):
                loss_dict[f"d{num_dec_layer}.loss_cls_dn"] = loss_cls_i
                loss_dict[f"d{num_dec_layer}.loss_pts_dn"] = loss_pts_i
                loss_dict[f"d{num_dec_layer}.loss_lclc_dn"] = loss_lclc_i
                num_dec_layer += 1

            num_dec_layer = 0
            for loss_segm_dice_i, loss_segm_mask_i in zip(dn_loss_segm_dice[:-1], dn_loss_segm_mask[:-1]):
                loss_dict[f"d{num_dec_layer}.loss_segm_dice_dn"] = loss_segm_dice_i
                loss_dict[f"d{num_dec_layer}.loss_segm_mask_dn"] = loss_segm_mask_i
                num_dec_layer += 1

        return loss_dict

    def calcul_dn_loss(self,
        gt_pts_list,
        preds_dicts,    
        ):
        (known_labels, known_masks, lclc_targets, pts_cls_scores, pts_preds, segm_mask_pred, lclc_preds, dn_group_num, num_tgt) = self.prepare_for_dn_loss(preds_dicts)
        num_dec_layers = segm_mask_pred.shape[0]

        pts = torch.cat(gt_pts_list).clone()
        known_pts = pts.repeat(dn_group_num, 1, 1).flatten(0, 1)
        all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]
        all_known_masks = [known_masks for _ in range(num_dec_layers)]

        dn_loss_segm_dice, dn_loss_segm_mask = multi_apply(
            self.dn_loss_single_segm, segm_mask_pred, all_known_masks, all_num_tgts_list
        )

        all_known_labels_list = [known_labels.long() for _ in range(len(pts_cls_scores))]
        all_known_pts_list = [known_pts for _ in range(len(pts_cls_scores))]
        all_lclc_target_list = [lclc_targets for _ in range(len(pts_cls_scores))]
        all_num_tgts_list = [num_tgt for _ in range(len(pts_cls_scores))]

        dn_loss_cls, dn_loss_pts, dn_loss_lclc = multi_apply(
            self.dn_loss_single_pts,
            pts_cls_scores,
            pts_preds,
            lclc_preds,
            all_known_labels_list,
            all_known_pts_list,
            all_lclc_target_list,
            all_num_tgts_list,
        )

        return dn_loss_segm_dice, dn_loss_segm_mask, dn_loss_cls, dn_loss_pts, dn_loss_lclc

    def dn_loss_single_pts(
        self,
        cls_scores,
        pts_preds,
        lclc_preds,
        known_labels,
        known_pts,
        known_lclc,
        num_total_pos=None,
        ):
        num_total_pos = cls_scores.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(num_total_pos, min=1.0).item()

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0

        cls_avg_factor = max(cls_avg_factor, 1)


        label_weights = torch.ones_like(known_labels)
        
        loss_cls = self.loss_cls(cls_scores, known_labels, label_weights, avg_factor=cls_avg_factor)
        isnotnan = torch.isfinite(known_pts).all(dim=-1)
        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        known_pts = known_pts.view(known_pts.size(0), -1 , self.pts_dim)
        pts_weights = torch.ones_like(known_pts)
        loss_pts = self.loss_bbox(
            pts_preds[isnotnan, :, :],
            known_pts[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )

        lclc_targets = 1 - known_lclc.view(-1).long()
        lclc_preds = lclc_preds.view(-1, 1)
        loss_lclc = self.lclc_branches[0].loss_rel(lclc_preds, lclc_targets)

        loss_cls = self.dn_weight * torch.nan_to_num(loss_cls)
        loss_pts = self.dn_weight * torch.nan_to_num(loss_pts)
        loss_lclc = self.dn_weight * torch.nan_to_num(loss_lclc)
        
        return loss_cls, loss_pts, loss_lclc

    def dn_loss_single_segm(
        self,
        segm_mask_preds,
        known_masks,
        num_total_pos=None,
        ):
        num_total_pos = known_masks.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(num_total_pos, min=1.0).item()

        segm_mask_shape = segm_mask_preds.shape[-2:]
        segm_mask_points_num = segm_mask_shape[0] * segm_mask_shape[1]

        if known_masks.shape[0] == 0:
            # zero match
            loss_dice = segm_mask_preds.sum()
            loss_mask = segm_mask_preds.sum()
            return loss_mask, loss_dice

        # dice loss
        loss_segm_dice = self.loss_segm_dice(segm_mask_preds, known_masks, avg_factor=num_total_pos)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        segm_mask_preds = segm_mask_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        known_masks = known_masks.reshape(-1)
        loss_segm_mask = self.loss_segm_mask(
            segm_mask_preds,
            known_masks,
            avg_factor=num_total_pos * segm_mask_points_num,
        )

        loss_segm_dice = self.dn_weight * torch.nan_to_num(loss_segm_dice)
        loss_segm_mask = self.dn_weight * torch.nan_to_num(loss_segm_mask)

        return loss_segm_dice, loss_segm_mask

    def prepare_for_dn_loss(self, mask_dict):

        (
            segm_mask_pred,
            pts_preds,
            pts_cls_scores,
            lclc_preds,
        ) = mask_dict["output_known_lbs_bboxes"]
        known_labels, known_masks = mask_dict["known_lbs_masks"]
        map_known_indice = mask_dict["map_known_indice"].long()
        known_indice = mask_dict["known_indice"].long()
        batch_idx = mask_dict["batch_idx"].long()
        bid = batch_idx[known_indice]
        lclc = torch.cat(mask_dict['lclc'], 0)
        dn_pad_size, dn_single_pad = mask_dict['pad_size'], mask_dict['dn_single_pad']
        dn_group_num = dn_pad_size // dn_single_pad
        lclc_targets_list = []
        lclc_preds_list = []
        for i in range(dn_group_num):
            lclc_preds_list.append(lclc_preds[:,:,i*dn_single_pad:(i+1)*dn_single_pad, i*dn_single_pad:(i+1)*dn_single_pad].flatten(2, 3))
            lclc_targets_list.append(lclc.flatten(0))
        lclc_preds = torch.cat(lclc_preds_list, 2)
        lclc_targets = torch.cat(lclc_targets_list, 0)
        num_tgt = known_indice.numel()
        if len(pts_cls_scores) > 0:
            num_layer, bs, num_q, num_p = pts_preds.shape
            pts_cls_scores = pts_cls_scores.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            pts_preds = pts_preds.view(num_layer, bs, num_q, -1, self.pts_dim).permute(1, 2, 0, 3, 4)[(bid, map_known_indice)].permute(1, 0, 2, 3)

            segm_mask_pred = segm_mask_pred.permute(1, 2, 0, 3, 4)[(bid, map_known_indice)].permute(1, 0, 2, 3)

        return known_labels, known_masks, lclc_targets, pts_cls_scores, pts_preds, segm_mask_pred, lclc_preds, dn_group_num, num_tgt
    
    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, rescale=False):


        num_query = preds_dicts['all_lanes_preds'].shape[2]

        lanes_preds = preds_dicts['all_lanes_preds'][-1]
        lanes_preds = lanes_preds.reshape(lanes_preds.shape[0],lanes_preds.shape[1],-1,self.pts_dim)

        o1_tensor = lanes_preds.unsqueeze(2).repeat(1, 1, num_query, 1,1)
        o2_tensor = lanes_preds.unsqueeze(1).repeat(1, num_query, 1, 1,1)
        topo = torch.sum(torch.abs(o1_tensor[:,:,:,-1,:]-o2_tensor[:,:,:,0,:]),dim=3)


        topo_mask = 1-torch.eye(num_query,device=topo.device)

        sigma = torch.std(topo)
        # print(sigma)
        # P = self.transformer.decoder.P.cpu()
        # w = self.transformer.decoder.w.cpu()
        # lamda_1 = float(self.transformer.decoder.lamda_1.cpu())
        # lamda_2 = float(self.transformer.decoder.lamda_2.cpu())

        P = 2
        w = 11.5275
        lamda_1 = 1
        lamda_2 = 1

        topo = torch.exp(-torch.pow(topo,P)/(w))*topo_mask

        distance_topo = topo.detach().cpu().numpy()
        Sim_topo = preds_dicts['all_lclc_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lclc_preds = lamda_1*Sim_topo + lamda_2*distance_topo

        
        all_lclc_preds = [_ for _ in all_lclc_preds]

        all_lcte_preds = preds_dicts['all_lcte_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lcte_preds = [_ for _ in all_lcte_preds]

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            lanes = preds['lane3d']
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([lanes, scores, labels])
        return ret_list, all_lclc_preds, all_lcte_preds
