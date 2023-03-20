from mimetypes import init
from re import S
from pandas import array

import torch
import torch.nn as nn
import cv2 as cv
import os
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from .lie_group_helper import make_c2w


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None

        if isinstance(init_c2w, str):
            poses = np.load(init_c2w).astype(np.float32)
            init_c2w = [torch.from_numpy(pose) for pose in poses]
            init_c2w = torch.stack(init_c2w)
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w


class LearnIntrin(nn.Module):
    def __init__(self, H, W, req_grad, fx_only=True, order=2, init_focal=None):
        super().__init__()
        self.H = H
        self.W = W
        self.order = order  # check our supplementary section.
        self.device = torch.device('cuda')

        if isinstance(init_focal, str):
            init_focal = np.load(init_focal)
        if init_focal is None:
            self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
        else:
            if self.order == 2:
                # a**2 * W = fx  --->  a**2 = fx / W
                coe_x = torch.sqrt(init_focal/ float(W)).clone().detach().float().requires_grad_(True)
                # coe_x = torch.tensor(torch.sqrt(init_focal / float(W)), requires_grad=False).float()
            elif self.order == 1:
                # a * W = fx  --->  a = fx / W
                coe_x = (init_focal/ float(W)).clone().detach().float().requires_grad_(True)
                coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
            else:
                print('Focal init order need to be 1 or 2. Exit')
                exit()
            self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )

    def forward(self, i=None):  # the i=None is just to enable multi-gpu training
        fx = self.fx.item()
        if self.order == 2:
            k = np.array([[fx**2 * self.W, 0., self.W/2, 0.],
                        [0., fx**2 * self.W, self.H/2, 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]], dtype=np.float32)
            # print(k.shape)
            # print(k)
            intrinsic = torch.from_numpy(k).to(self.device)
        else:
            k = np.array([[fx * self.W, 0., self.W/2, 0.],
                        [0., fx * self.W, self.H/2, 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]], dtype=np.float32)
            intrinsic = torch.from_numpy(k).to(self.device)

        return intrinsic
      

class RaysGenerator:
    def __init__(self, img_lis, msk_lis, depth_lis, pose_net, intrin_net,
                 learnable=False, with_depth=False):
        super(RaysGenerator, self).__init__()
        # self.image_size = image_size
        self.pose_net = pose_net
        self.intrin_net = intrin_net
        self.learnable = learnable
        self.with_depth = with_depth

        if not learnable:
             self.intrin_inv = torch.inverse(self.intrin_net)
        print('Load data: Begin')
        self.device = torch.device('cuda')
        
        self.images_lis = img_lis
        self.n_images = len(self.images_lis)
        
        self.images_np = np.stack([cv.imread(im_name, -1) for im_name in self.images_lis]) / 255.0
        if self.images_np.shape[-1]==4:
            print('[Debug]here with rgba images')
            pic_data, a = self.images_np[:,:,:,:3], self.images_np[:,:,:,3:]
            pic_data = pic_data*a + (1-a)  # white
            # pic_data = pic_data*a  # black
            self.images_np = pic_data
            self.masks_np = a
        else:
            print('[Debug] read mask from file')
            self.masks_lis = msk_lis
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0
            self.images_np = self.images_np*self.masks_np+(1-self.masks_np)
        # print(self.masks_np.max(),self.masks_np.min())
        # print(self.images_np.max(),self.images_np.min())
        # print(self.n_images)
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        image_size = self.images.shape[1:3]
        if with_depth:
            self.depth_lis = depth_lis
            self.up_feats = nn.Upsample(size=image_size, mode='bilinear')
            self.depths_np = np.stack([np.squeeze(np.load(fname)) for fname in self.depth_lis])
            m, s = np.mean(self.depths_np), np.std(self.depths_np)
            self.depths_np = (self.depths_np-m)/s
            print(self.depths_np.max(), self.depths_np.min())
            self.depth_feats = torch.sigmoid(torch.from_numpy(self.depths_np.astype(np.float32)).cpu())
            if self.depth_feats.dim()==3:
                self.depth_feats = self.depth_feats.unsqueeze(1)
            self.depth_feats = self.up_feats(self.depth_feats)
            self.depth_feats = self.depth_feats.permute(0,2,3,1)
            print(self.depth_feats.shape)
            assert self.images.shape[:-1]==self.depth_feats.shape[:-1], 'self.images in {} doesnot match self.depth_feats in {}'.format(self.images.shape[:-1], self.depth_feats.shape[:-1])
        # assert self.images.shape[1:3]==self.image_size, 'self.images in {} doesnot match self.image_size in {}'.format(self.images.shape, self.image_size)
        print(self.images.shape)
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W
        print('Load data: End')

    # kb cropping
    def cropping(self, img, crop_size=(352, 1216), resolution_level=1):
        h_im, w_im = img.shape[:2]
        
        margin_top = max(0, int(h_im - crop_size[0]))
        margin_left = max(0, int((w_im - crop_size[1])/ 2))
        margin_bottom = min(h_im, margin_top + crop_size[0])
        margin_right = min(w_im, margin_left + crop_size[1])

        img = img[margin_top: margin_bottom,
                  margin_left: margin_right]
        img = cv.resize(img, (crop_size[1] // resolution_level, crop_size[0] // resolution_level))
        return img


    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        if self.learnable:
            pose = self.pose_net(img_idx)
            intrinsic_inv = torch.inverse(self.intrin_net())
        else:
            pose = self.pose_net[img_idx]
            intrinsic_inv = self.intrin_inv[img_idx]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)
        
    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        if self.learnable:
            pose = self.pose_net(img_idx)
            intrinsic_inv = torch.inverse(self.intrin_net())
        else:
            pose = self.pose_net[img_idx]
            intrinsic_inv = self.intrin_inv[img_idx]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        # print(intrinsic_inv[None, :3, :3].shape, p.shape)  # 1, 4, 4
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        feats = torch.zeros([rays_o.shape[0],1]).cpu()
        if self.with_depth:
            feats = self.depth_feats[img_idx][(pixels_y, pixels_x)]    # batch_size, 1
        return torch.cat([rays_o.cpu(), rays_v.cpu(), mask[:, :1], color, feats], dim=-1).cuda()  # batch_size, 10
        
    def gen_rays_between(self, ratio, idx_0, idx_1, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        if self.learnable:
            pose_0 = self.pose_net(idx_0).detach().cpu().numpy()
            pose_1 = self.pose_net(idx_1).detach().cpu().numpy()
            intrinsic_inv = torch.inverse(self.intrin_net()).detach().cpu().numpy()
        else:
            pose_0 = self.pose_net[idx_0]
            pose_1 = self.pose_net[idx_1]
            intrinsic_inv = self.intrin_inv[0]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3

        trans = pose_0[:3, 3] * (1.0 - ratio) + pose_1[:3, 3] * ratio
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def image_at(self, idx, resolution_level):
        img = self.images_np[idx]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))*255).clip(0, 255)
        
    def mask_at(self, idx, resolution_level):
        msk = self.masks_np[idx]
        # return cv.resize(msk, (self.W // resolution_level, self.H // resolution_level))
        return np.expand_dims(cv.resize(msk, (self.W // resolution_level, self.H // resolution_level)), axis=-1)
