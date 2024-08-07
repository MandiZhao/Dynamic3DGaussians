from scipy.spatial import ConvexHull, Delaunay
import copy
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
import json
import os
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Delaunay
import glob
import h5py
import math
from argparse import ArgumentParser

def upsample_mesh(vertices, faces, colors=None, num_samples_per_triangle=3):
    """
    Upsamples a mesh given its vertices, faces, and optional vertex colors.

    :param vertices: A numpy array of shape (N, 3) representing mesh vertices.
    :param faces: A numpy array of shape (3, M) representing indices of vertices forming each triangular face.
    :param colors: Optional. A numpy array of shape (N,) representing colors at each vertex. Defaults to None.
    :param num_samples_per_triangle: Number of points to sample per triangle.
    :return: A tuple (upsampled_points, upsampled_colors) where:
        upsampled_points is a numpy array of the new points on the mesh.
        upsampled_colors is a numpy array of the colors for these points.
    """
    # Create a list to store all upsampled points and colors
    all_sampled_points = []
    all_sampled_colors = []

    for face in faces.T:
        # Extract the vertices for the current face
        triangle_vertices = vertices[face]

        # Generate random barycentric coordinates for the samples
        barycentric_coords = np.random.dirichlet([1, 1, 1], size=num_samples_per_triangle)

        # Compute the cartesian coordinates of the samples
        sampled_points = np.dot(barycentric_coords, triangle_vertices)
        all_sampled_points.append(sampled_points)

        if colors is not None:
            # Extract the colors for the vertices of the current face
            face_colors = colors[face]

            # Interpolate the colors for the samples
            sampled_colors = np.dot(barycentric_coords, face_colors.reshape(-1, 3))
            all_sampled_colors.append(sampled_colors)

    # Combine all samples into a single array
    upsampled_points = np.vstack(all_sampled_points)

    if colors is not None:
        upsampled_colors = np.vstack(all_sampled_colors)
    else:
        upsampled_colors = None

    return upsampled_points, upsampled_colors


def sample_gaussian_means(triangle_vertices, barycentric_coords, num_samples_per_triangle):
    # add one dimension to the baricenters and repeat for each coordinate of the vertices
    barycentric_coords = np.expand_dims(barycentric_coords, axis=3)
    barycentric_coords = np.repeat(barycentric_coords, 3, axis=3)
    # add one dimension to the vertices and repeat for each num_sample_per_triange
    triangle_vertices = np.expand_dims(triangle_vertices, axis=0)
    triangle_vertices = np.repeat(triangle_vertices, num_samples_per_triangle, axis=0)
    # get the sample points as multiplitcation of the barycentric coordinates and the vertices
    sampled_points = np.einsum('ijkm,ijkm->ijkm', barycentric_coords, triangle_vertices)
    sampled_points = sampled_points.sum(axis=-2)
    return sampled_points
def get_mesh(points, plot=False):
    points2d = points[:, :2]
    # Compute the Delaunay triangulation
    tri = Delaunay(points2d)
    if plot:
        # Visualize the triangulation
        plt.triplot(points2d[:, 0], points2d[:, 1], tri.simplices)
        plt.plot(points2d[:, 0], points2d[:, 1], 'o')
        plt.show()

def compute_edges_index(points, k=3, delaunay=False, sim_data=False, norm_threshold=0.01):
    if delaunay:
        if sim_data:
            points2d = points[:, [0, 2]]
        else:
            points2d = points[:, :2]
        tri = Delaunay(points2d)
        edges = set()
        faces = []
        for simplex in tri.simplices:
            valid_face = True
            current_edges = []

            for i in range(3):
                p1, p2 = simplex[i], simplex[(i + 1) % 3]
                edge = (min(p1, p2), max(p1, p2))
                current_edges.append(edge)
                # Calculate the norm (distance) between the points
                norm = np.linalg.norm(points2d[p1] - points2d[p2])

                # Check if the edge meets the threshold condition
                if norm_threshold is not None and norm > norm_threshold:
                    valid_face = False
                else:
                    edges.add(edge)

            # Add the face if all edges are valid
            if valid_face:
                faces.append(simplex)

        edge_index = np.asarray(list(edges))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Convert faces list to a tensor
        faces = torch.tensor(np.asarray(faces), dtype=torch.long).t().contiguous()
        return edge_index, faces
    else:
        # Use a k-D tree for efficient nearest neighbors computation
        tree = cKDTree(points)
        # For simplicity, we find the 3 nearest neighbors; you can adjust this number
        _, indices = tree.query(points, k=k + 1)

        # Skip the first column because it's the point itself
        edge_index = np.vstack({tuple(sorted([i, j])) for i, row in enumerate(indices) for j in row[1:]})
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index


def load_mesh(path):
    with h5py.File(path, 'r') as f:
        data = np.asarray(f['pos'])

    return data

def get_upsampled_points(init_mesh_path, num_samples_per_triangle=3):
    graph_pos = load_mesh(init_mesh_path)

    _, faces = compute_edges_index(graph_pos, k=3, delaunay=True, sim_data=False, norm_threshold=0.1)
    upsampled_points, upsampled_colors = upsample_mesh(graph_pos, faces, colors=None, num_samples_per_triangle=num_samples_per_triangle)

    return graph_pos, upsampled_points

def plot_mesh(vertices, upsampled_points):
    # Plotting
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot initial mesh points
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], marker='o', s=50, alpha=0.6,
                label='Initial Points')
    ax1.set_title('Initial Mesh Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot upsampled points
    ax2.scatter(upsampled_points[:, 0], upsampled_points[:, 1], upsampled_points[:, 2], marker='o',
                s=20, alpha=0.6, label='Upsampled Points')
    ax2.set_title('Upsampled Mesh Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Setting plot limits for better comparison
    # x_limits = np.concatenate((vertices[:, 0], upsampled_points[:, 0]))
    # y_limits = np.concatenate((vertices[:, 1], upsampled_points[:, 1]))
    # z_limits = np.concatenate((vertices[:, 2], upsampled_points[:, 2]))

    for ax in [ax1, ax2]:
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_zlim([-0.3, 0.3])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the mesh
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/alberta/Downloads/SHORTS_01_00/splits/init_mesh.hdf5',nargs='+')
    parser.add_argument('--num_samples_per_triangle', type=int, default=10)
    parser.add_argument('--output',type=str,default=None)
    args = parser.parse_args()


    paths = args.path
    
    for path in paths:

        if os.path.isdir(path):
            path = glob.glob(os.path.join(path, '*.hdf5'))[0]
            print(path)

        # increase num samples per triangle to get more points
        num_samples_per_triangle = args.num_samples_per_triangle
        graph_pos, upsampled_points = get_upsampled_points(path, num_samples_per_triangle=num_samples_per_triangle)

        if args.output is not None:
            output = args.output
        else:
            output = os.path.join(os.path.dirname(path), 'init_pt_cld.npz')
        print("saving {} points to {}".format(upsampled_points.shape[0], output))   
        np.savez(output, upsampled_points=upsampled_points)
        # plot to debug number of points needed
        plot_mesh(graph_pos, upsampled_points)