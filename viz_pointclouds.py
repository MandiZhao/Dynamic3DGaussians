from argparse import ArgumentParser 
import numpy as np 
import matplotlib.pyplot as plt
import open3d as o3d

parser = ArgumentParser()
parser.add_argument('--gt', type=str, default='')
parser.add_argument('--means3D', type=str, default='')
args = parser.parse_args()


gt_traj = np.load(args.gt)['pos']
gt_t0 = gt_traj[0] # N x 3
pred_traj = np.load(args.means3D) # N x 3
if args.means3D.endswith(".npz"):
    pred_traj = pred_traj['upsampled_points']

# show both point clouds in 3D using open3d

gt_pcd = o3d.geometry.PointCloud()
gt_pcd.points = o3d.utility.Vector3dVector(gt_t0)
gt_pcd.paint_uniform_color([0, 0, 1])

pred_pcd = o3d.geometry.PointCloud()
pred_pcd.points = o3d.utility.Vector3dVector(pred_traj)
pred_pcd.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries([gt_pcd, pred_pcd])


