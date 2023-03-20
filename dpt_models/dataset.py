import torch
# import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
# from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf, image_size=(896, 1184), with_depth=False):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.img_dir = conf.get_string('img_dir')
        self.depth_dir = conf.get_string('depth_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        self.image_size = image_size
        self.images_lis = sorted(glob(os.path.join(self.data_dir, self.img_dir, '*.png')))
        self.n_images = len(self.images_lis)
        print('[Info] find {} images in img_folder {}'.format(self.n_images, os.path.join(self.data_dir, self.img_dir)))
        # self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = [os.path.join(self.data_dir, self.img_dir,  'mask', '{}.png'.format(os.path.basename(fim)[:-4])) for fim in self.images_lis]
        self.depth_lis = [os.path.join(self.data_dir, self.img_dir,  self.depth_dir, '{}.npy'.format(os.path.basename(fim)[:-4])) for fim in self.images_lis]
        self.H, self.W = cv.imread(self.images_lis[0]).shape[:2]
        self.image_pixels = self.H * self.W

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_{}'.format((os.path.basename(idx)[:-4]))].astype(np.float32) for idx in self.images_lis]

        self.scale_mats_np = []
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_{}'.format((os.path.basename(idx)[:-4]))].astype(np.float32) for idx in self.images_lis]
        # for idx in self.images_lis:
        #     scale_mat = camera_dict['scale_mat_{}'.format((os.path.basename(idx)[:-4]))].astype(np.float32)
        #     tmp = -scale_mat[2,3]
        #     scale_mat[2,3] = scale_mat[1,3]
        #     scale_mat[1,3] = tmp
        #     self.scale_mats_np.append(scale_mat)
        
        # for trainning in LearnPoses& LearnIntrin
        # camera_infos = np.load(os.path.join(img_folder, 'RT_mats.npy'))
        # camera_infos = np.load(os.path.join(img_folder, 'w2c_mats.npy'))
        # self.intrinsics_all = [camera_infos['intrinsic_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        # self.pose_all = [camera_infos['pose_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        # object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_scale_mat = self.scale_mats_np[0]
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        # print(mid)
        near = mid - 1.0
        far = mid + 1.0
        return near, far
