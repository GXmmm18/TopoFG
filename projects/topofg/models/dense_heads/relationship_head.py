import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.models.builder import HEADS, build_loss


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@HEADS.register_module()
class LLRelationshipHead(nn.Module):
    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 pts_dim=None,
                 shared_param=False,
                 num_layer=2,
                 output_dim=128,
                 loss_rel=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25)):
        super().__init__()

        self.MLP_o1 = MLP(in_channels_o1+pts_dim, in_channels_o1//2, output_dim, num_layer)
        self.shared_param = shared_param
        if shared_param:
            self.MLP_o2 = self.MLP_o1
        else:
            self.MLP_o2 = MLP(in_channels_o2+pts_dim, in_channels_o2//2, output_dim, num_layer)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim*2, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, 1),
        )
        self.loss_rel = build_loss(loss_rel)

    def forward(self, end_feats, start_feats, dn_pad_size=0):
        # feats: B, num_query, num_embedding
        bs, num_q, dim = end_feats.shape
        end_embeds = self.MLP_o1(end_feats).unsqueeze(2)
        start_embeds = self.MLP_o2(start_feats).unsqueeze(1)
        if dn_pad_size != 0:
            dn_end_embeds = end_embeds[:, :dn_pad_size]
            dn_start_embeds = start_embeds[:, :, :dn_pad_size]
            dn_real_embeding = torch.cat([dn_end_embeds.expand(-1, -1, dn_pad_size, -1), dn_start_embeds.expand(-1, dn_pad_size, -1, -1)], dim=-1)
            dn_relationship_pred = self.classifier(dn_real_embeding)

            end_embeds = end_embeds[:, dn_pad_size:]
            start_embeds = start_embeds[:, :, dn_pad_size:]
            num_q = num_q-dn_pad_size

        real_embeding = torch.cat([end_embeds.expand(-1, -1, num_q, -1), start_embeds.expand(-1, num_q, -1, -1)], dim=-1)
        relationship_pred = self.classifier(real_embeding)

        if dn_pad_size == 0:
            dn_relationship_pred = relationship_pred[:,:dn_pad_size,]


        return relationship_pred, dn_relationship_pred



@HEADS.register_module()
class LTRelationshipHead(nn.Module):
    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=False,
                 num_layer=2,
                 output_dim=256,
                 loss_rel=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25)):
        super().__init__()

        self.MLP_o1 = MLP(in_channels_o1*2, in_channels_o1, output_dim, num_layer)
        self.shared_param = shared_param
        if shared_param:
            self.MLP_o2 = self.MLP_o1
        else:
            self.MLP_o2 = MLP(in_channels_o2*2, in_channels_o2, output_dim, num_layer)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim*2, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, 1),
        )
        self.loss_rel = build_loss(loss_rel)
        # self.loss_rel = FocalLossWithLogitMargin()

    def forward(self, lane_feats, lane_coords, te_feats, te_cls_scores):
        # feats: B, num_query, num_embedding
        bs, num_ql, _, dim = lane_feats.shape
        num_qt = te_feats.shape[1]
        lane_vec = (lane_coords[:, :, -4, :2] - lane_coords[:, :, 3, :2]).squeeze(0)
        main_dir = lane_vec.new_tensor((1, 0)).unsqueeze(0).repeat(num_ql, 1)
        dot = (lane_vec * main_dir).sum(dim=1)
        cos_theta = torch.clamp(dot / (lane_vec.norm(dim=1) * main_dir.norm(dim=1)  + 1e-8), -1.0, 1.0)  # [N]
        angles = torch.acos(cos_theta)
        angle_weight = angles.new_tensor(np.pi) - angles
        angle_weight = angle_weight.unsqueeze(0).unsqueeze(-1).repeat(1, 1, dim)
        
        scores, _ = te_cls_scores.sigmoid().max(-1)
        scores = scores*2
        te_weight = scores.unsqueeze(-1).repeat(1, 1, dim)

        lane_embeds = self.MLP_o1(torch.cat([lane_feats.mean(2), angle_weight], dim=-1)).unsqueeze(2)
        te_embeds = self.MLP_o2(torch.cat([te_feats, te_weight], dim=-1)).unsqueeze(1)

        real_embeding = torch.cat([lane_embeds.expand(-1, -1, num_qt, -1), te_embeds.expand(-1, num_ql, -1, -1)], dim=-1)
        relationship_pred = self.classifier(real_embeding)

        return relationship_pred
    
    def loss(self, rel_preds, gt_adjs, o1_assign_results, o2_assign_results):
        B, num_query_o1, num_query_o2, _ = rel_preds.size()
        o1_assign = o1_assign_results
        o1_pos_inds = o1_assign['pos_inds']
        o1_pos_assigned_gt_inds = o1_assign['pos_assigned_gt_inds']

        if self.shared_param:
            o2_assign = o1_assign
            o2_pos_inds = o1_pos_inds
            o2_pos_assigned_gt_inds = o1_pos_assigned_gt_inds
        else:
            o2_assign = o2_assign_results
            o2_pos_inds = o2_assign['pos_inds']
            o2_pos_assigned_gt_inds = o2_assign['pos_assigned_gt_inds']

        targets = []
        for i in range(B):
            gt_adj = gt_adjs[i]
            target = torch.zeros_like(rel_preds[i].squeeze(-1), dtype=gt_adj.dtype, device=rel_preds.device)
            xs = o1_pos_inds[i].unsqueeze(-1).repeat(1, o2_pos_inds[i].size(0))
            ys = o2_pos_inds[i].unsqueeze(0).repeat(o1_pos_inds[i].size(0), 1)
            target[xs, ys] = gt_adj[o1_pos_assigned_gt_inds[i]][:, o2_pos_assigned_gt_inds[i]]
            targets.append(target)
        targets = torch.stack(targets, dim=0)

        targets = 1 - targets.view(-1).long()
        rel_preds = rel_preds.view(-1, 1)

        loss_rel = self.loss_rel(rel_preds, targets)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_rel = torch.nan_to_num(loss_rel)

        return loss_rel
