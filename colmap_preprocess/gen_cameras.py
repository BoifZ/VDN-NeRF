import numpy as np
import trimesh
import cv2 as cv
import sys
import os
from glob import glob


if __name__ == '__main__':
    work_dir = sys.argv[1]
    # work_dir += '/dense/0'

    from pose_utils import load_colmap_data
    _, _, perm, names = load_colmap_data(work_dir)
    names = names[perm]

    poses_hwf = np.load(os.path.join(work_dir, 'poses.npy')) # n_images, 3, 5
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    # hwf = hwf*1200/4032
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]
    pts = []
    pts.append((pose @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([1, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 1, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 0, 1, 1])[:, None]).squeeze()[:3])
    pts = np.stack(pts, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(work_dir, 'pose.ply'))
    #

    cam_dict = dict()
    n_images = len(poses_raw)

    # Convert space
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    print(n_images)
    for i, name in enumerate(names):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = (w - 1) * 0.5
        intrinsic[1, 2] = (h - 1) * 0.5
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        # cam_dict['camera_mat_{:03d}'.format(int(name[:-4]))] = intrinsic
        # cam_dict['camera_mat_inv_{:03d}'.format(int(name[:-4]))] = np.linalg.inv(intrinsic)
        # cam_dict['world_mat_{:03d}'.format(int(name[:-4]))] = world_mat
        # cam_dict['world_mat_inv_{:03d}'.format(int(name[:-4]))] = np.linalg.inv(world_mat)
        cam_dict['camera_mat_{:0>3d}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{:0>3d}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{:0>3d}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{:0>3d}'.format(i)] = np.linalg.inv(world_mat)


    pcd = trimesh.load(os.path.join(work_dir, 'sparse_points_interest.ply'))
    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center
    print(scale_mat)

    # for i in range(n_images):
    for i, name in enumerate(names):
        # cam_dict['scale_mat_{:03d}'.format(int(name[:-4]))] = scale_mat
        # cam_dict['scale_mat_inv_{:03d}'.format(int(name[:-4]))] = np.linalg.inv(scale_mat)
        cam_dict['scale_mat_{:0>3d}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{:0>3d}'.format(i)] = np.linalg.inv(scale_mat)

    # out_dir = os.path.join(work_dir, 'preprocessed')
    # os.makedirs(out_dir, exist_ok=True)
    # os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    # os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

    # image_list = glob(os.path.join(work_dir, 'preprocessed/image/*[.png]'))
    # image_list.sort()

    # print('work_dir', work_dir,'images #', len(image_list))
    # train_res = (1920, 1080)
    # for i, image_path in enumerate(image_list):
    #     img = cv.imread(image_path)
    #     # print(img.shape)
    #     # H, W, _ = img.shape
    #     img = cv.resize(img, train_res)
    #     cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)
    #     cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), np.ones_like(img) * 255)

    np.savez(os.path.join(work_dir, 'cameras_sphere_colmap.npz'), **cam_dict)
    print('Process done!')


# python export_inlier_pairs.py --dense_folder <PATH> --save_folder <PATH>
# root = r'E:\aaboif\datasets\overall\fin1\ZIL--\2000'
# with open(os.path.join(root, 'images_name.txt'), 'r') as file:
#     names = file.read().splitlines()

# camd = {}
# d = np.load(os.path.join(root, 'cameras_sphere_mid.npz'))
# for f in d.files:
#     camd[f] = d[f]
# d = np.load(os.path.join(root, 'cameras_sphere_up.npz'))
# for f in d.files:
#     camd[f+' (2)'] = d[f]
# d = np.load(os.path.join(root, 'cameras_sphere_low.npz'))
# for f in d.files:
#     camd[f+' (3)'] = d[f]

# cam_dict = {}
# for idx, n in enumerate(names):
#     cam_dict['world_mat_{:0>3d}'.format(idx)] = camd['world_mat_{}'.format(n[:-4])]
#     cam_dict['camera_mat_{:0>3d}'.format(idx)] = camd['camera_mat_{}'.format(n[:-4])]
#     cam_dict['scale_mat_{:0>3d}'.format(idx)] = camd['scale_mat_{}'.format(n[:-4])]
# np.savez(os.path.join(root, 'cameras_sphere_gt.npz'), **cam_dict)