import open3d as o3d
import os
from os.path import join
import numpy as np
import json
from glob import glob
from natsort import natsorted
from PIL import Image
import matplotlib.pyplot as plt

def get_pointcloud(depth, rgb, extrinsic, intrinsic):
    img_h = depth.shape[0]
    img_w = depth.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x, pixel_y = np.meshgrid(
        np.linspace(0, img_w - 1, img_w), np.linspace(0, img_h - 1, img_h)
    )
    cam_pts_x = np.multiply(
        pixel_x - intrinsic[0, 2], depth / intrinsic[0, 0]
    )
    cam_pts_y = np.multiply(
        pixel_y - intrinsic[1, 2], depth / intrinsic[1, 1]
    )
    cam_pts_z = depth 
    cam_pts = (
        np.array([cam_pts_x, cam_pts_y, cam_pts_z])
        .transpose(1, 2, 0)
        .reshape(-1, 3)
    )
    world_pts = np.matmul(
        extrinsic,
        np.concatenate((cam_pts, np.ones_like(cam_pts[:, [0]])), axis=1).T,
    ).T[:, :3]
    
    xyz_pts = world_pts.astype(np.float32)
    rgb_pts = rgb[:, :, :3].reshape(-1, 3)
    return xyz_pts, rgb_pts

def get_extrinsics(transform_matrix):
    P = transform_matrix.copy()
    R = P[:3,:3]
    t = P[:3,3]
    P_inv = np.eye((4))
    P_inv[:3,:3] = R.T
    P_inv[:3,3] = -R.T@t
    P_inv[2,:] *= -1
    P_inv[1,:] *= -1
    return P_inv

INP_PATH = "../corl_1_dense_rgb"
folder = "train"
loaded_depth = np.load(
    join(INP_PATH, "depth_first.npz")
    # join(INP_PATH, folder, 'depth.npz')
    )
depth_fnames = loaded_depth['filenames']
depth_data = loaded_depth['depth']
pcds = []
seg_pcds = []
tstep = 0
num_tsteps = 41
for view_i in range(5): 
    rgb_idx = view_i * num_tsteps + tstep
    depth_idx = view_i 
    depth = depth_data[depth_idx]
    print(depth_fnames[depth_idx])
    
    depth[depth > 20] = 20 
    imgs = natsorted( glob(join(INP_PATH, folder, "*png")) )
    img = imgs[rgb_idx]
    print('rgb:', img)
    rgbd = Image.open(img)
    h, w = rgbd.size
    json_fname = join(INP_PATH, f"transforms_{folder}.json")
    with open(json_fname, 'r') as f:
        transforms = json.load(f)

    angle_x, angle_y = transforms["camera_angle_x"], transforms["camera_angle_y"]
    focal_x = (h / 2) / np.tan(angle_x / 2)
    focal_y = (w / 2) / np.tan(angle_y / 2)
    intrin = np.array([[focal_x, 0.0, h / 2], [0.0, focal_y, w / 2], [0.0, 0.0, 1.0]]) 

    assert transforms['frames'][rgb_idx]["file_path"].split('/')[-1] == imgs[rgb_idx].split('/')[-1]

    trans_matrix = np.array(transforms['frames'][rgb_idx]["transform_matrix"])  
    
    extrin = get_extrinsics(trans_matrix) 

    img = o3d.io.read_image(imgs[rgb_idx])  # 4 channels
    color = np.array(np.asarray(img)[:, :, :3]).astype('uint8') 

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(color), 
        depth=o3d.geometry.Image(depth.astype('float32')),
        depth_scale=1, depth_trunc=20, convert_rgb_to_intensity=False)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(w, h, focal_x, focal_y, w / 2, h / 2)
    intrinsic_matrix = np.array(
        [
            [focal_x, 0.0, h / 2],
            [0.0, focal_y, w / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd, intrinsic=intrinsic, extrinsic=extrin)

    seg = np.asarray(img)[:, :, 3]
    # convert to 3 channel  
    seg_rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    # convert non-zero pixels to red 
    seg_rgb[seg > 0] = [255,255,255] 

    seg_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(seg_rgb.astype('uint8')),
        depth=o3d.geometry.Image(depth.astype('float32')),
        depth_scale=1, depth_trunc=5, convert_rgb_to_intensity=False)
    seg_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=seg_rgbd, intrinsic=intrinsic, extrinsic=extrin)
    if np.array(seg_pcd.points).shape[0] == np.array(pcd.points).shape[0]:
        print("same number of points")
        pcds.append(pcd)
        seg_pcds.append(seg_pcd) 

print("Visualize merged point clouds")
merged_pcd = o3d.geometry.PointCloud()
for pcd in pcds:
    merged_pcd += pcd
merged_seg_pcd = o3d.geometry.PointCloud()
for seg_pcd in seg_pcds:
    merged_seg_pcd += seg_pcd

# save to npz
points = np.array(merged_pcd.points)
colors = np.array(merged_pcd.colors) # should be 0-1!
seg_colors = np.array(merged_seg_pcd.colors)
# convert from (N,3) to (N,1) binary
segs = np.array([0 if np.all(seg == [0,0,0]) else 1 for seg in seg_colors])

tosave_array = np.concatenate((points, colors, segs[:,None]), axis=1)

downsample_size = 500000
sub_idxs = np.random.choice(len(tosave_array), downsample_size, replace=(len(tosave_array) < downsample_size))
tosave_array = tosave_array[sub_idxs]
print(tosave_array.shape) # should be N, 7
np.savez_compressed(join(INP_PATH, folder, "init_pt_cld.npz"), data=tosave_array) # this gives 'data' key

# visualizer = o3d.visualization.Visualizer()
# visualizer.create_window()
# visualizer.add_geometry(merged_pcd)
# visualizer.poll_events()
# visualizer.update_renderer()
# visualizer.run()
breakpoint()



