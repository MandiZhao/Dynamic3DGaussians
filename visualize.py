
import numpy as np
import open3d as o3d
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
import argparse
from os.path import join
import imageio

RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'depth'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'trajectories'  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'rotations'  # None, 'trajectories' or 'rotations'

REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True

w, h = 640, 360
# NEW:
w, h = 800, 800 
near, far = 0.01, 100.0
view_scale = 3.9
fps = 20
traj_frac = 25  # 4% of points
traj_length = 15
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()


def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def load_scene_data(seq, exp, seg_as_col=False):
    # params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    param_fnams = glob(f"{DATA_DIR}/output/{exp}/{seq}/params_*.npz")
    param_fnams = natsorted(param_fnams)
    print(f"Loading {param_fnams[-1]}")
    params = dict(np.load(param_fnams[-1]))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()} 
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def calculate_trajectories(scene_data, is_fg, traj_frac=200, traj_length=1):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    num_lines = len(in_pts[0])
    cols = np.repeat(colormap[np.arange(len(in_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    out_pts = []
    for t in range(len(in_pts))[traj_length:]:
        out_pts.append(np.array(in_pts[t - traj_length:t + 1]).reshape(-1, 3))
    # breakpoint()
    return make_lineset(out_pts, cols, num_lines)


def calculate_rot_vec(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    in_rotation = [data['rotations'][is_fg][::traj_frac] for data in scene_data]
    num_lines = len(in_pts[0])
    cols = colormap[np.arange(num_lines) % len(colormap)]
    inv_init_q = deepcopy(in_rotation[0])
    inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
    inv_init_q = inv_init_q / (inv_init_q ** 2).sum(-1)[:, None]
    init_vec = np.array([-0.1, 0, 0])
    out_pts = []
    for t in range(len(in_pts)):
        cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
        rot = build_rotation(cam_rel_qs).cpu().numpy()
        vec = (rot @ init_vec[None, :, None]).squeeze()
        out_pts.append(np.concatenate((in_pts[t] + vec, in_pts[t]), 0))
    return make_lineset(out_pts, cols, num_lines)


def render(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    d_near = 1.5
    d_far = 6
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

def load_test_camera(fname="data/corl_1_dense_rgb/val_meta.json"):
    with open(fname, 'r') as f:
        meta = json.load(f)
    w2cs = np.array(meta['w2c'])
    ks = np.array(meta['k'])
    return w2cs, ks

def visualize(seq, exp):
    scene_data, is_fg = load_scene_data(seq, exp) 
    
    # w2c, k = init_camera()
    # load test time camera:
    w2cs, ks = load_test_camera(fname="data/corl_1_dense_rgb/val_meta.json")
    w2c = w2cs[0][0]
    k = ks[0][0] 
    im, depth = render(w2c, k, scene_data[0])
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == 'depth'))
    # save image
    Image.fromarray(
        np.array(
            im.cpu().permute(1, 2, 0) * 255, dtype=np.uint8
            )
            ).save(f"./output/{exp}/{seq}/init.png")
    # breakpoint()
    o3d.visualization.webrtc_server.enable_webrtc()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    linesets = None
    lines = None
    if ADDITIONAL_LINES is not None:
        if ADDITIONAL_LINES == 'trajectories':
            linesets = calculate_trajectories(scene_data, is_fg, traj_frac=traj_frac, traj_length=1)
        elif ADDITIONAL_LINES == 'rotations':
            linesets = calculate_rot_vec(scene_data, is_fg)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    view_k = k * view_scale
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(h * view_scale)
    cparams.intrinsic.width = int(w * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = view_scale
    render_options.light_on = False

    start_time = time.time()
    num_timesteps = len(scene_data)
    render_count = 0
    while True:
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps
        if ADDITIONAL_LINES == 'trajectories':
            t = int(passed_frames % (num_timesteps - traj_length)) + traj_length  # Skip t that don't have full traj.
        else:
            t = int(passed_frames % num_timesteps)

        if FORCE_LOOP:
            num_loops = 1.4
            y_angle = 360*t*num_loops / num_timesteps
            w2c, k = init_camera(y_angle)
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            cam_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        else:  # Interactive control
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / view_scale
            k[2, 2] = 1
            w2c = cam_params.extrinsic 
        if RENDER_MODE == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data[t]['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data[t]['colors_precomp'].contiguous().double().cpu().numpy())
        else:

            im, depth = render(w2c, k, scene_data[t])
            print("rendered step:", t, im.max(), im.mean())
            # if os.path.exists(f"./output/{exp}/{seq}/{t}.png"):
            breakpoint()
            Image.fromarray(
            np.array(
                im.cpu().permute(1, 2, 0) * 255, dtype=np.uint8
                )
                ).save(f"./output/{exp}/{seq}/{t}.png")
            render_count += 1
            pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == 'depth'))
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if ADDITIONAL_LINES is not None:
            if ADDITIONAL_LINES == 'trajectories':
                lt = t - traj_length
            else:
                lt = t
            lines.points = linesets[lt].points
            lines.colors = linesets[lt].colors
            lines.lines = linesets[lt].lines
            vis.update_geometry(lines)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options

def render_video(seq, exp, meta_fname="data/corl_1_dense_rgb/val_meta.json", use_cameras=20, fps=20):
    import imageio
    scene_data, is_fg = load_scene_data(seq, exp) 
    run_dir = f"{DATA_DIR}/output/{exp}/{seq}"
    w2cs, ks = load_test_camera(fname=meta_fname) # shape tstep, n_cameras, 4, 4
    all_renders = []
    n_frames = min(w2cs.shape[0], len(scene_data))
    n_cameras = w2cs.shape[1]
    cam_idxs = np.random.choice(n_cameras, use_cameras, replace=False)
    for cam in cam_idxs:
        curr_frame = []
        for t in range(n_frames):
            w2c = w2cs[t][cam]
            k = ks[t][cam] 
            im, depth = render(w2c, k, scene_data[t])
            rgb = im.cpu().permute(1, 2, 0).numpy()
            # breakpoint()
            rgb =  (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            curr_frame.append(rgb)
        all_renders.extend(
            curr_frame
        )
    
    # save video
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    if 'train' in meta_fname:
        video_fname = "train_cameras.mp4"
    else:
        video_fname = "test_cameras.mp4"
    
    video_fname = join(run_dir, video_fname)
    imageio.mimwrite(video_fname, all_renders, fps=fps, quality=8)
    print(f"Saved video to {video_fname}")
    return 


def plot_traj(seq, exp, totrack_pts=None, gt_traj=None):
    scene_data, is_fg = load_scene_data(seq, exp)  
    run_dir = f"{DATA_DIR}/output/{exp}/{seq}"
    linesets = calculate_trajectories(scene_data, is_fg, traj_frac=200, traj_length=1)
    # plot the linesets in 3D
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    if totrack_pts is None:
        # plot all the pts
        num_trajs = len(linesets[0].lines) 
        colors = colormap[np.arange(num_trajs) % len(colormap)]
        for lineset in linesets:
            points = np.array(lineset.points)
            lines = np.array(lineset.lines) # (N, 2)
            for i, line in enumerate(lines):
                start, end = line # (2,)
                color = colors[i]
                ax.plot(points[[start, end], 0], points[[start, end], 1], points[[start, end], 2], color=color, alpha=0.9)
    else:
        # find the closest GSs for each point 
        init_points = scene_data[0]['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() 
        colors = colormap[np.arange(len(totrack_pts)) % len(colormap)]
        for i, pt in enumerate(totrack_pts):
            dists = np.linalg.norm(init_points - pt, axis=1)
            closest = np.argmin(dists)
            # breakpoint()
            closest_dist = dists[closest]
            # print("closest_dist:", closest_dist)
            color = colors[i]
            for j, lineset in enumerate(linesets):
                points = np.array(lineset.points)
                lines = np.array(lineset.lines)
                start, end = lines[closest]
                ax.plot(points[[start, end], 0], points[[start, end], 1], points[[start, end], 2], color=color, alpha=0.9)
                # scatter the start and end points
                # if j == 0:
                #     ax.scatter(
                #         points[start, 0], points[start, 1], points[start, 2], color='red', label="start")
                # if j == len(linesets)-1:
                #     ax.scatter(
                #         points[end, 0], points[end, 1], points[end, 2], color='blue', label="end")
    plt.tight_layout()
    fname = f"traj_track_{len(totrack_pts)}pts.png" if totrack_pts is not None else f"traj_track_allpts.png"
    
    def save_plot_video(ax, fname):
        imgs = []
        for az in range(0, 360, 60):
            ax.view_init(elev=20., azim=az)
            view_fname = f"v{az}_{fname}"
            view_fname = join(run_dir, view_fname)
            plt.savefig(view_fname, dpi=300)
            imgs.append(Image.open(view_fname))
            
        # save mp4 
        video_fname = fname.replace(".png", ".mp4")
        video_fname = join(run_dir, video_fname)
        imageio.mimwrite(video_fname, imgs, fps=2, quality=8)
    
    save_plot_video(ax, fname)

    if gt_traj is not None:
        ax = fig.add_subplot(111, projection='3d')
        num_pts = gt_traj.shape[1]
        for i, pt in enumerate(gt_traj[0]):
            color = colormap[i % len(colormap)]
            traj = gt_traj[:,i] # (time, 3)
            ax.plot(traj[:,0], traj[:,1], traj[:,2], color=color, alpha=0.9)
        fname = f"traj_gt_{len(totrack_pts)}pts.png" 
        save_plot_video(ax, fname)
        fname = join(run_dir, fname)
        plt.savefig(fname, dpi=400)
 
    breakpoint()
if __name__ == "__main__":
    # exp_name = "pretrained"
    # for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    #     visualize(sequence, exp_name)
    # visualize('corl_1_dense_rgb', "subsample_100k")
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--track', action='store_true', default=False)
    parser.add_argument('--exp_name', '-ex', type=str, default="subsample_100k")
    parser.add_argument('--data', '-d', type=str, default="corl_1_dense_rgb")
    parser.add_argument('--load_gt_fname', default="../tracking_utils/gt_trajs/corl_1_dense_traj.npz")
    parser.add_argument('--track_num_pts', '-n', type=int, default=100)
    parser.add_argument('--track_all', action='store_true', default=False)
    parser.add_argument('--fps', type=int, default=15)
    args = parser.parse_args()

    if args.render:
        render_video(
            args.data, args.exp_name, 
            meta_fname=f"{DATA_DIR}/data/corl_1_dense_rgb/val_meta.json",
            use_cameras=20, fps=args.fps)
        
    
    traj_length = 1 # this is so dumb
    if args.track:
        traj_frac = 2000 #4000
        load_gt_fname = args.load_gt_fname
        num_pts = args.track_num_pts
        gt_traj = np.load(load_gt_fname)['traj'] # ~20k points, shape (t, N, 3)
        idxs = np.random.choice(gt_traj.shape[1], num_pts, replace=False)
        totrack_pts = gt_traj[0][idxs] # (N, 3)
        gt_traj = gt_traj[:, idxs] # (t, N, 3)  

        plot_traj(args.data, args.exp_name, totrack_pts, gt_traj)
    
    if args.track_all:
        traj_frac = 200 #4000 
        totrack_pts = None
        gt_traj = None

        plot_traj(args.data, args.exp_name, totrack_pts, gt_traj)

        
    
    
    
    
