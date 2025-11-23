#---------------------------------------------------------------------------------------#
# Graph-based Topology Reasoning for Driving Scenes (https://arxiv.org/abs/2304.05277)  #
# Source code: https://github.com/OpenDriveLab/TopoNet                                  #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#
import cv2
import numpy as np
from shapely import affinity
import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString
from ...core.lane.util import fix_pts_interpolate


@PIPELINES.register_module()
class LaneParameterize3D(object):

    def __init__(self, method, method_para):
        method_list = ['fix_pts_interp']
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        lanes = results['gt_lanes_3d']
        para_lanes = getattr(self, self.method)(lanes, **self.method_para)
        results['gt_lanes_3d'] = para_lanes

        return results

    def fix_pts_interp(self, input_data, n_points=11):
        '''Interpolate the 3D lanes to fix points. The input size is (n_pts, 3).
        '''
        lane_list = []
        for lane in input_data:
            if n_points == 11 and lane.shape[0] == 201:
                lane_list.append(lane[::20].flatten())
            else:
                lane = fix_pts_interpolate(lane, n_points).flatten()
                lane_list.append(lane)
        return np.array(lane_list, dtype=np.float32)


@PIPELINES.register_module()
class LaneLengthFilter(object):
    """Filter the 3D lanes by lane length (meters).
    """

    def __init__(self, min_length):
        self.min_length = min_length

    def __call__(self, results):

        if self.min_length <= 0:
            return results

        length_list = np.array(list(map(lambda x:LineString(x).length, results['gt_lanes_3d'])))
        masks = length_list > self.min_length
        results['gt_lanes_3d'] = [lane for idx, lane in enumerate(results['gt_lanes_3d']) if masks[idx]]
        results['gt_lane_labels_3d'] = results['gt_lane_labels_3d'][masks]

        if 'gt_lane_adj' in results.keys():
            results['gt_lane_adj'] = results['gt_lane_adj'][masks][:, masks]
        if 'gt_lane_lcte_adj' in results.keys():
            results['gt_lane_lcte_adj'] = results['gt_lane_lcte_adj'][masks]

        return results



@PIPELINES.register_module()
class GenerateLaneMask(object):

    def __init__(self, mask_size=[100, 200], patch_size=[50, 100], n_pts_per_vec=11):

        self.mask_size = mask_size
        self.patch_size = patch_size
        self.scale_y = self.mask_size[0] / self.patch_size[0]
        self.scale_x = self.mask_size[1] / self.patch_size[1]
        self.n_pts_per_vec = n_pts_per_vec

    def __call__(self, results):
        """Call function to create instance mask.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: mask results, 'gt_masks_list' key is added into
                result dict.
        """
        lanes = results['gt_lanes_3d']
        num_gt = lanes.shape[0]
        gt_masks_list = self.instance_segments_condi(lanes.reshape(num_gt, self.n_pts_per_vec, -1)[...,:2])
        results['gt_masks_list'] = gt_masks_list
        return results

    def instance_segments_condi(self, lanes, thickness=3):

        assert len(lanes) != 0
        instance_segm_list = []
        for instance in lanes:
            instance_segm = np.zeros((self.mask_size[0], self.mask_size[1]), dtype=np.uint8)
            try:
                self.line_ego_to_mask(LineString(instance), instance_segm, color=1, thickness=thickness)
            except:
                pass
            instance_segm_list.append(instance_segm)
        instance_segm_list = np.stack(instance_segm_list)
        return instance_segm_list

    def line_ego_to_mask(self, line_ego, mask, color=1, thickness=2):
        """Rasterize a single line to mask.

        Args:
            line_ego (LineString): line
            mask (array): semantic mask to paint on
            color (int): positive label, default: 1
            thickness (int): thickness of rasterized lines, default: 3
        """
        trans_x = self.mask_size[1] / 2
        trans_y = self.mask_size[0] / 2
        line_ego = affinity.scale(line_ego, self.scale_x, self.scale_y, origin=(0, 0))
        line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])

        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        coords[:, 1] = mask.shape[0] - coords[:, 1]
        assert len(coords) >= 2
        cv2.polylines(mask, np.int32([coords]), False, color=color, thickness=thickness)
