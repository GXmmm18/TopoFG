import numpy as np
import torch
from shapely.geometry import LineString

def normalize_3dlane(lanes, pc_range):
    normalized_lanes = lanes.clone()
    normalized_lanes[..., 0::3] = (lanes[..., 0::3] - pc_range[0]) / (pc_range[3] - pc_range[0])
    normalized_lanes[..., 1::3] = (lanes[..., 1::3] - pc_range[1]) / (pc_range[4] - pc_range[1])
    normalized_lanes[..., 2::3] = (lanes[..., 2::3] - pc_range[2]) / (pc_range[5] - pc_range[2])
    normalized_lanes = torch.clamp(normalized_lanes, 0, 1)

    return normalized_lanes

def denormalize_3dlane(normalized_lanes, pc_range):
    lanes = normalized_lanes.clone()
    lanes[..., 0::3] = (normalized_lanes[..., 0::3] * (pc_range[3] - pc_range[0]) + pc_range[0])
    lanes[..., 1::3] = (normalized_lanes[..., 1::3] * (pc_range[4] - pc_range[1]) + pc_range[1])
    lanes[..., 2::3] = (normalized_lanes[..., 2::3] * (pc_range[5] - pc_range[2]) + pc_range[2])
    return lanes

def fix_pts_interpolate(lane, n_points):
    ls = LineString(lane)
    distances = np.linspace(0, ls.length, n_points)
    lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane

def sample_coords(mask, ins_query_emb, pc_range, patch_size, num_samples=11):

    bs, num_mask, h, w = mask.shape
    dtype = ins_query_emb.dtype
    x_scale = patch_size[0] / w
    y_scale = patch_size[1] / h
    coordinates_list = []
    for b in range(bs):
        coordinates_batch = []
        for m in range(num_mask):
            mask_single = mask[b, m, :, :]  
            coordinates = torch.stack(torch.where(mask_single == 1), dim=-1)  # [N, 2]
            if coordinates.size(0) > 0:
                coordinates[:, 0] = h - 1 - coordinates[:, 0]
                coordinates[:, 0] = coordinates[:, 0] - h/2
                coordinates[:, 1] = coordinates[:, 1] - w/2
                coordinates_sorted = coordinates[coordinates[:, 0].argsort()]
                coordinates_sorted = coordinates_sorted[coordinates_sorted[:, 1].argsort()]
                coordinates_sorted[:, [0, 1]] = coordinates_sorted[:, [1, 0]]
                N = coordinates_sorted.size(0)
                if N > num_samples:
                    sampled_indices = torch.linspace(0, N - 1, num_samples, dtype=torch.long, device=mask.device)
                    sampled_coords = coordinates_sorted[sampled_indices.long()]
                else:
                    sampled_indices = torch.linspace(0, N - 1, num_samples, dtype=torch.long, device=mask.device)
                    sampled_coords = coordinates_sorted[sampled_indices.long() % N]
            else:
                y = torch.linspace(0, h - 1, num_samples, dtype=torch.long, device=mask.device)
                x = torch.linspace(w // 4, 3 * w // 4, num_samples, dtype=torch.long, device=mask.device)
                y = h - 1 - y
                y = y - h // 2
                x = x - w // 2 
                sampled_coords = torch.stack([x, y], dim=-1)

            output_coords = sampled_coords.to(dtype)
            coordinates_batch.append(output_coords)

        coordinates_list.append(torch.stack(coordinates_batch, dim=0))
    
    
    coordinates_list = torch.stack(coordinates_list, dim=0)
    
    z_coords = torch.zeros_like(coordinates_list[..., 0:1])
    coordinates_list = torch.cat([coordinates_list, z_coords], dim=-1)
        

    coordinates_list[..., 0] = coordinates_list[..., 0] * x_scale
    coordinates_list[..., 1] = coordinates_list[..., 1] * y_scale
    bs, num_p, _, pt_dim = coordinates_list.shape
    coordinates_list = normalize_3dlane(coordinates_list.flatten(2, 3), pc_range)

    return coordinates_list.view(bs, num_p, -1, pt_dim)
