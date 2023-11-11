import os
import numpy as np
import torch
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
from copy import deepcopy
from PIL import Image
import os
from glob import glob 
from natsort import natsorted 
import json
from train import DATA_DIR
from visualize import load_scene_data, calculate_trajectories
import argparse
from os.path import join
import imageio
from external import build_rotation # this is the q2R function mentioned in paper
from einops import rearrange
from colormap import colormap
import shutil

def compute_influence(inp_pt, opacities, scales, rotations, means3D):
    sig_opacity = torch.sigmoid(opacities) # N, 1
    num_gs = sig_opacity.shape[0]
    device = sig_opacity.device # should be on "cuda"
    if type(inp_pt) == np.ndarray:
        inp_pt = torch.from_numpy(inp_pt)
    # cast to float32
    if inp_pt.dtype != torch.float32:
        inp_pt = inp_pt.float()
    inp_pt = inp_pt.to(device)
 
    cov_diag = torch.zeros((num_gs, 3, 3), device=device) # -> S
    # set diagonal entries # N, 3
    cov_diag[:, 0, 0] = scales[:, 0]
    cov_diag[:, 1, 1] = scales[:, 1]
    cov_diag[:, 2, 2] = scales[:, 2]

    cov_q = rotations  # N, 4
    cov_rot = build_rotation(cov_q) # N, 3, 3 -> R
    # R@S@S^T@R^T
    s_T = rearrange(cov_diag, 'n i j -> n j i')
    r_T = rearrange(cov_rot, 'n i j -> n j i')
    cov_matrix = cov_rot @ cov_diag @ s_T @ r_T # careful  CUDA out of memory.
    inv_cov = torch.inverse(cov_matrix) # N, 3, 3
    centers = means3D # N, 3
    offset = inp_pt - centers # N, 3
    offset = rearrange(offset, 'n i -> n i ()') # N, 3, 1
    offset_T = rearrange(offset, 'n i j -> n j i') # N, 1, 3

    tosig = torch.exp(-0.5 * offset_T @ inv_cov @ offset) # N, 1, 1
    tosig = rearrange(tosig, 'n i j -> n (i j)') # N, 1
    influence_p = sig_opacity * tosig  
    return influence_p

exp_name = "50k_full"
traj_frac = 200
scene_data, is_fg = load_scene_data("corl_1_dense_rgb", exp_name)
linesets = calculate_trajectories(scene_data, is_fg, traj_frac=traj_frac, traj_length=1)
# {('opacities', torch.Size([198081, 1])), ('colors_precomp', torch.Size([198081, 3])), ('rotations', torch.Size([198081, 4])), ('means2D', torch.Size([198081, 3])), ('scales', torch.Size([198081, 3])), ('means3D', torch.Size([198081, 3]))}
# 
load_gt_fname = "../tracking_utils/gt_trajs/corl_1_dense_traj.npz"
first_frame_data = scene_data[0] 
opacities = first_frame_data['opacities'][::traj_frac]
scales = first_frame_data['scales'][::traj_frac]
rotations = first_frame_data['rotations'][::traj_frac]
means3D = first_frame_data['means3D'][::traj_frac]

num_pts = 150
gt_traj = np.load(load_gt_fname)['traj'] # ~20k points, shape (t, N, 3)
idxs = np.random.choice(gt_traj.shape[1], num_pts, replace=False)
totrack_pts = gt_traj[0][idxs] # (N, 3)
gt_traj = gt_traj[:, idxs] # (t, N, 3)  

from matplotlib import pyplot as plt
fig = plt.figure()
ax_influence = fig.add_subplot(131, projection='3d')
ax_dist = fig.add_subplot(132, projection='3d')
ax_gt = fig.add_subplot(133, projection='3d')

colors = colormap[np.arange(len(totrack_pts)) % len(colormap)]
tracked_traj_influence = []
tracked_traj_dist = []
for i, pt in enumerate(totrack_pts):
    influence_p = compute_influence(
        pt, opacities, scales, rotations, means3D
        )
    highest_idx = torch.argmax(influence_p)
    # print("highest influence point", influence_p[highest_idx])
    traj_influence = []
    color = colors[i]
    for j, lineset in enumerate(linesets):
        points = np.array(lineset.points)
        lines = np.array(lineset.lines)
        start, end = lines[highest_idx]
        ax_influence.plot(points[[start, end], 0], points[[start, end], 1], points[[start, end], 2], color=color)
        if j == 0:
            traj_influence.append(points[start, :])
        traj_influence.append(points[end, :])
    tracked_traj_influence.append(traj_influence)

    # try plot the distance only
    dists = np.linalg.norm(means3D.detach().cpu().numpy() - pt, axis=1)
    low_idx = np.argmin(dists)
    # print("lowest distance point", dists[highest_idx])
    traj_dist = []
    for j, lineset in enumerate(linesets):
        points = np.array(lineset.points)
        lines = np.array(lineset.lines)
        start, end = lines[low_idx]
        ax_dist.plot(points[[start, end], 0], points[[start, end], 1], points[[start, end], 2], color=color)
        if j == 0:
            traj_dist.append(points[start, :])
        traj_dist.append(points[end, :])
    tracked_traj_dist.append(traj_dist)

    # also plot the GT trajectory
    ax_gt.plot(gt_traj[:, i, 0], gt_traj[:, i, 1], gt_traj[:, i, 2], color=color)

tracked_traj_influence = np.array(tracked_traj_influence) # (N, t, 3)
tracked_traj_dist = np.array(tracked_traj_dist) # (N, t, 3)
# transpose to (t, N, 3)
tracked_traj_influence = np.transpose(tracked_traj_influence, (1, 0, 2))
tracked_traj_dist = np.transpose(tracked_traj_dist, (1, 0, 2))
save_fname = f"track_data/infl/{exp_name}.npz"
np.savez(save_fname, traj=tracked_traj_influence) # save the tracked trajectory
save_fname = f"track_data/dist/{exp_name}.npz"
np.savez(save_fname, traj=tracked_traj_dist)
shutil.copy(load_gt_fname, f"track_data/dist/gt.npz")
shutil.copy(load_gt_fname, f"track_data/infl/gt.npz")
# save the tracked trajectory

fname = f"tracking_{exp_name}.png"  
ax_influence.set_title("Highest Influence")
ax_dist.set_title("Lowest Distance")
ax_gt.set_title("Ground Truth")
# [ax.axis('off') for ax in [ax_influence, ax_dist, ax_gt]]
plt.tight_layout()
plt.savefig(fname, dpi=500)
breakpoint()
