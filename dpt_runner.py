import os
import sys
from re import L, T
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from dpt_models.dataset import Dataset, load_K_Rt_from_P
from dpt_models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from dpt_models.renderer import NeuSRenderer
from dpt_models.poses import LearnPose, LearnIntrin, RaysGenerator


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', img_dir='image', npz_postfix='', is_continue=False, latest_ckpt=None):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        conf_text = conf_text.replace('IMG_DIR', img_dir)
        conf_text = conf_text.replace('TYPE', npz_postfix)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        # self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        # self.conf['dataset.img_dir'] = self.conf['dataset.img_dir'].replace('IMG_DIR', img_dir)
        # self.conf['dataset.render_cameras_name'] = self.conf['dataset.render_cameras_name'].replace('TYPE', (npz_postfix))
        # self.conf['dataset.object_cameras_name'] = self.conf['dataset.object_cameras_name'].replace('TYPE', npz_postfix)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        if img_dir != 'image':
            self.base_exp_dir += '_' + img_dir.split('image')[-1]
        print(self.base_exp_dir)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0
        self.poses_iter_step = 0
        self.depth_iter = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_int('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_int('train.anneal_end', default=0.0)

        self.learnable = self.conf.get_bool('train.focal_learnable', default=False)
        self.extract_depth = self.conf.get_bool('train.extract_depth', default=False)
        self.fix_depthnet = False
        if self.extract_depth:
            self.only_depth = self.conf.get_bool('train.only_depth')
            self.depth_before_color =  self.conf.get_bool('train.depth_before_color')
            self.depth_start_iter = self.conf.get_int('train.depth_start_iter')
            # self.fix_depthnet = True
        else:
            self.only_depth = self.depth_before_color = False
            
        self.rgb_dims = self.conf.get_int('train.rgb_dims') if self.extract_depth else 3
        if self.learnable:
            self.focal_lr = self.conf.get_float('train.focal_lr')
            self.pose_lr = self.conf.get_float('train.pose_lr')
            self.focal_lr_gamma = self.conf.get_float('train.focal_lr_gamma')
            self.pose_lr_gamma = self.conf.get_float('train.pose_lr_gamma')
            self.step_size = self.conf.get_int('train.step_size')

            self.start_refine_pose_iter = self.conf.get_int('train.start_refine_pose_iter')
            self.start_refine_focal_iter = self.conf.get_int('train.start_refine_focal_iter')

            # learn focal parameter
            self.intrin_net = LearnIntrin(self.dataset.H, self.dataset.W, **self.conf['model.focal'], init_focal=self.dataset.focal).to(self.device)
            # learn pose for each image
            self.pose_param_net = LearnPose(self.dataset.n_images, **self.conf['model.pose'], init_c2w=self.dataset.pose_all).to(self.device)
            self.optimizer_focal = torch.optim.Adam(self.intrin_net.parameters(), lr=self.focal_lr)
            self.optimizer_pose = torch.optim.Adam(self.pose_param_net.parameters(), lr=self.pose_lr)

            self.scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_focal, milestones=(self.warm_up_end, self.end_iter, self.step_size),
                                                                gamma=self.focal_lr_gamma)
            self.scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_pose, milestones=range(self.warm_up_end, self.end_iter, self.step_size),
                                                                gamma=self.pose_lr_gamma)
        else:
            self.intrin_net = self.dataset.intrinsics_all
            self.pose_param_net = self.dataset.pose_all
  
        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.use_mask = self.conf.get_bool('train.use_mask', default=False)
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None
        if not 'mesh' in self.mode:
            self.rays_generator = RaysGenerator(self.dataset.images_lis, self.dataset.masks_lis, self.dataset.depth_lis, 
                                                self.pose_param_net, self.intrin_net, # image_size=(800,800),  # (1080, 1920),  ## (896, 1184), # (900, 1200),
                                                learnable=self.learnable, with_depth=self.extract_depth)

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        if self.extract_depth:
            print('depth together with renderer')
            # add depth_feats
            self.depth_weight = self.conf.get_float('train.depth_weight')
            self.depth_network = RenderingNetwork(**self.conf['model.depth_extract_network']).to(self.device)
            params_to_train += list(self.depth_network.parameters())
            if self.fix_depthnet:
                for p in self.depth_network.parameters():
                    p.requires_grad=False
        else:
            self.depth_network = None

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.depth_network,
                                     **self.conf['model.neus_renderer'])

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name.startswith('ckpt') and model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        if self.fix_depthnet:
            return
        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def depth_iter_weight(self, total_iter=5000):
        # weight = self.depth_iter/total_iter
        # if weight > 1: return 1.0
        weight = 1./(np.exp(-10*(self.depth_iter/total_iter-0.5))+1.)
        return  weight

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        
        if self.learnable:
            if self.poses_iter_step >= self.start_refine_pose_iter:
                self.pose_param_net.train()
            else:
                self.pose_param_net.eval()

            if self.poses_iter_step >= self.start_refine_focal_iter:
                self.intrin_net.train()
            else:
                self.intrin_net.eval()
        flag = True
        for iter_i in tqdm(range(res_step)):
            if self.learnable:
                if self.poses_iter_step >= self.start_refine_pose_iter:
                    self.pose_param_net.train()
                if self.poses_iter_step >= self.start_refine_focal_iter:
                    self.intrin_net.train()

            img_idx = image_perm[self.iter_step % len(image_perm)]
            # data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
            data = self.rays_generator.gen_random_rays_at(img_idx, self.batch_size)
            # print(data.shape)
            rays_o, rays_d, mask, true_rgb, gt_feats = data[:, :3], data[:, 3: 6], data[:, 6: 7], data[:, 7: 7+self.rgb_dims], data[:, 7+self.rgb_dims:]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.use_mask:
                mask = (mask > 0.1).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              depth_before_color=self.depth_before_color)

            depth_feats = render_out['render_feats']
            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight
                
            if self.extract_depth and self.iter_step > self.depth_start_iter and not self.fix_depthnet:
                depth_feat_error = (depth_feats - gt_feats) * mask
                depth_fine_loss = F.l1_loss(depth_feat_error, torch.zeros_like(depth_feat_error), reduction='sum') / mask_sum
                psnr_dfeat = 20.0 * torch.log10(1.0 / (((depth_feats - gt_feats)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())
                loss += depth_fine_loss * self.depth_iter_weight()
                self.writer.add_scalar('Loss/depth_loss', depth_fine_loss, self.iter_step)
                self.writer.add_scalar('Statistics/psnr_dfeat', psnr_dfeat, self.iter_step)
                # self.optimizer_depth.zero_grad()
                self.depth_iter += 1

            self.optimizer.zero_grad()
            if self.learnable and self.iter_step > self.start_refine_pose_iter:
                self.optimizer_focal.zero_grad()
                self.optimizer_pose.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.learnable and self.iter_step > self.start_refine_pose_iter:
                self.optimizer_focal.step()
                self.optimizer_pose.step()

            self.iter_step += 1
            self.poses_iter_step += 1
            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))
                if self.extract_depth and self.iter_step > self.depth_start_iter and not self.fix_depthnet:
                    print(' depth_loss = {}'.format(depth_fine_loss))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                res = 128
                world = False
                if self.iter_step % 150000==0: 
                    res = 512
                    world = True
                elif self.iter_step % 50000==0: 
                    res = 256
                self.validate_mesh(world_space=world, resolution=res)

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()
            
            # if self.iter_step == 20000:
            #     self.val_all_imgs(resolution_level=2, gen_depth_for_finetune=True)
            #     print('val_imgs_done. gen_sdf_feats for finetune')

        self.val_all_imgs(resolution_level=2, gen_depth_for_finetune=False, both_mask=True)
        
    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

        if self.learnable:
            self.scheduler_focal.step()
            self.scheduler_pose.step()

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        base_folder = os.path.dirname(sys.argv[0])
        print('[Debug] base_folder: ', base_folder)
        for dir_name in dir_lis:
            source_dir = os.path.join(base_folder, dir_name)
            print('[Info file_backup]', source_dir)
            if os.path.isfile(source_dir):
                cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
                if dir_name[-3:] == '.py':
                    copyfile(source_dir, cur_dir)
                continue
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(source_dir)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(source_dir, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))
        with open(os.path.join(self.base_exp_dir, 'recording', 'config.conf'), 'a+') as f:
            f.seek(0)
            f.write('# '+' '.join(sys.argv)+'\n')

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'], strict=False)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        if self.extract_depth and self.iter_step > self.depth_start_iter:
            self.depth_network.load_state_dict(checkpoint['depth_network_fine'])
        
        if self.learnable and self.iter_step > self.start_refine_pose_iter:
            self.load_pnf_checkpoint(checkpoint_name.replace('ckpt', 'pnf'))

        logging.info('End')

    def save_checkpoint(self, prefix='ckpt'):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'depth_network_fine': self.depth_network.state_dict() if self.extract_depth else None,
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)

        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', '{}_{:0>6d}.pth'.format(prefix, self.iter_step)))
        if self.learnable:
            self.save_pnf_checkpoint()

    def load_pnf_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'pnf_checkpoints', checkpoint_name), map_location=self.device)
        self.intrin_net.load_state_dict(checkpoint['intrin_net'])
        self.pose_param_net.load_state_dict(checkpoint['pose_param_net'])
        self.optimizer_focal.load_state_dict(checkpoint['optimizer_focal'])
        self.optimizer_pose.load_state_dict(checkpoint['optimizer_pose'])
        self.poses_iter_step = checkpoint['poses_iter_step']

    def save_pnf_checkpoint(self):
        pnf_checkpoint = {
            'intrin_net': self.intrin_net.state_dict(),
            'pose_param_net': self.pose_param_net.state_dict(),
            'optimizer_focal': self.optimizer_focal.state_dict(),
            'optimizer_pose': self.optimizer_pose.state_dict(),
            'poses_iter_step': self.poses_iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'pnf_checkpoints'), exist_ok=True)
        torch.save(pnf_checkpoint, os.path.join(self.base_exp_dir, 'pnf_checkpoints', 'pnf_{:0>6d}.pth'.format(self.iter_step)))

    def store_current_pose(self):
        self.pose_net.eval()
        num_cams = self.pose_net.module.num_cams if isinstance(self.pose_net, torch.nn.DataParallel) else self.pose_net.num_cams

        c2w_list = []
        for i in range(num_cams):
            c2w = self.pose_net(i)  # (4, 4)
            c2w_list.append(c2w)
            
        c2w_list = torch.stack(c2w_list)  # (N, 4, 4)
        c2w_list = c2w_list.detach().cpu().numpy()
        np.save(os.path.join(self.base_exp_dir, 'cam_poses', 'pose_{:0>6d}.npy'.format(self.iter_step)), c2w_list)
        return
    
    def val_img(self, idx, resolution_level=1, gen_depth_for_finetune=False, both_mask=False):
        gt = self.rays_generator.image_at(idx, resolution_level=resolution_level)/255.0
        mask = self.rays_generator.mask_at(idx, resolution_level=resolution_level)
        # self.use_mask = True
        if self.use_mask or both_mask:
            mask = (mask > 0.1).astype(np.float32)
        else:
            mask = np.ones_like(mask)
        
        # rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        rays_o, rays_d = self.rays_generator.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        gradient_error = []
        out_weight_depth = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                            rays_d_batch,
                                            near,
                                            far,
                                            cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                            background_rgb=background_rgb,
                                            depth_before_color=self.depth_before_color)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            gradient_error.append(render_out['gradient_error'].detach().cpu().numpy())
            if gen_depth_for_finetune:
                weights = render_out['weights']
                inside_sphere = render_out['inside_sphere']
                weights = weights[:,:inside_sphere.shape[1]]*inside_sphere
                weight_max = torch.argmax(weights, dim=-1).view(rays_o_batch.shape[0],-1).detach()  # .cpu().numpy()
                z_vals = render_out['z_vals'].detach()
                weight_depth = z_vals.gather(dim=1, index=weight_max).cpu()
                out_weight_depth.append(weight_depth)
            del render_out

        if gen_depth_for_finetune:
            fname = os.path.basename(self.rays_generator.images_lis[idx])
            out_weight_depth = np.concatenate(out_weight_depth, axis=0).reshape([H, W, -1])
            os.makedirs(os.path.join(self.dataset.data_dir, self.dataset.img_dir, 'depth_from_sdf'), exist_ok=True)
            np.save(os.path.join(self.dataset.data_dir, self.dataset.img_dir, 'depth_from_sdf', 'sdf_{}.npy'.format(fname[:-4])), out_weight_depth)
            # out_weight_depth[out_weight_depth>0.1]=0
            # out_weight_depth *= 2550
            # out_weight_depth[out_weight_depth<20] = 150
            mma, mmi = out_weight_depth.max(), out_weight_depth.min()
            lb, ub = np.percentile(out_weight_depth, [50, 95])
            # lb, ub = 6, 8
            # print(mma, mmi, lb, ub)
            out_weight_depth = ((out_weight_depth - lb)/(ub-lb)*255).clip(0, 255)
            os.makedirs(os.path.join(self.base_exp_dir, 'weight_max'), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'weight_max', 'weight_max_{}_{}.png'.format(self.iter_step, idx)), out_weight_depth)
        

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]))
        mask_sum = mask.sum() + 1e-5
        color_loss = torch.from_numpy((img_fine - gt)*mask)
        color_fine_loss = F.l1_loss(color_loss, torch.zeros_like(color_loss), reduction='sum') / mask_sum
        psnr = 20.0 * np.log10(1.0 / np.sqrt(((img_fine - gt)**2 * mask).sum() / (mask_sum * 3.0)))

        color_fine_loss1=None
        psnr1=None
        if both_mask:
            mask = np.ones_like(mask)
            mask_sum = mask.sum() + 1e-5
            color_loss1 = torch.from_numpy((img_fine - gt)*mask)
            color_fine_loss1 = F.l1_loss(color_loss1, torch.zeros_like(color_loss1), reduction='sum') / mask_sum
            psnr1 = 20.0 * np.log10(1.0 / np.sqrt(((img_fine - gt)**2 * mask).sum() / (mask_sum * 3.0)))

        return color_fine_loss, psnr, gradient_error, color_fine_loss1, psnr1

    def val_all_imgs(self, resolution_level=-1, gen_depth_for_finetune=False, both_mask=False):
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        closses = []
        psnrs = []
        closses1 = []
        psnrs1 = []
        glosses = []
        for idx in range(self.dataset.n_images):
            if idx%10==0: print(idx)
            color_fine_loss, psnr, gradient_error, color_fine_loss1, psnr1 = self.val_img(idx, resolution_level=resolution_level, gen_depth_for_finetune=gen_depth_for_finetune, both_mask=both_mask)
            closses.append(color_fine_loss)
            psnrs.append(psnr)
            if both_mask:
                closses1.append(color_fine_loss1)
                psnrs1.append(psnr1)
            glosses.append(gradient_error)
        closses = np.stack(closses)
        psnrs = np.stack(psnrs)
        glosses = np.stack(glosses)
        print(np.mean(closses), np.mean(psnrs), np.mean(glosses))
        if both_mask:
            closses1 = np.stack(closses1)
            psnrs1 = np.stack(psnrs1)
            print(np.mean(closses1), np.mean(psnrs1))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.rays_generator.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              depth_before_color=self.depth_before_color)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, self.rgb_dims, -1]) * 255).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        # print(img_fine.shape)
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                val_im = np.concatenate([img_fine[..., i],
                                           self.rays_generator.image_at(idx, resolution_level=resolution_level)])
                cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           val_im)
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.rays_generator.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              depth_before_color=self.depth_before_color)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine
    
    def get_gt_poses(self, cameras_sphere, cam_num, color=None, length=0.5):
        from vis_cam_traj import draw_camera_frustum_geometry

        if color is None:
            color = np.random.rand(1, 3)
        camera_dict = np.load(cameras_sphere)
        intrinsics_all = []
        pose_all = []
        for idx in range(cam_num):
            scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
            world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(intrinsics.astype(np.float32))
            pose_all.append(pose.astype(np.float32))

        c2w_gt = np.array(pose_all)
        fx_gt = intrinsics_all[0][0, 0]
        # gt_color = np.array([color], dtype=np.float32)
        # gt_color = np.tile(gt_color, (cam_num, 1))
        gt_est_list = draw_camera_frustum_geometry(c2w_gt, self.dataset.H, self.dataset.W,
                                                        fx_gt, fx_gt,
                                                        length, color)
        return gt_est_list
    
    def show_cam_pose(self, iter_show=-1, random_color=True):
        import open3d as o3d
        from vis_cam_traj import draw_camera_frustum_geometry

        cam_num = 33
        # cam_num = self.dataset.n_images

        '''Get focal'''
        fxfy = self.intrin_net(0).cpu().detach().numpy()[0][0]
        print('learned cam intrinsics:')
        print('fxfy', fxfy)

        '''Get all poses in (N, 4, 4)'''
        c2ws_est = torch.stack([self.pose_param_net(i) for i in range(cam_num)])  # (N, 4, 4)

        '''Frustum properties'''
        frustum_length = 0.5
        random_color = random_color
        all_color = np.random.rand(3, 3)
        if random_color:
            frustum_color = np.random.rand(cam_num, 3)
        else:
            # frustum_color = np.array([[249, 65, 68]], dtype=np.float32) / 255
            frustum_color = np.array([all_color[0]], dtype=np.float32)
            frustum_color = np.tile(frustum_color, (cam_num, 1))

        '''Get frustums'''
        frustum_est_list = draw_camera_frustum_geometry(c2ws_est.cpu().detach().cpu().numpy(), self.dataset.H, self.dataset.W,
                                                        fxfy, fxfy,
                                                        frustum_length, frustum_color)
        
        # init poses
        c2w_init = self.dataset.pose_all
        fx_init = self.dataset.focal.cpu().detach()
        init_color = np.array([all_color[1]], dtype=np.float32)
        init_color = np.tile(init_color, (cam_num, 1))
        frustum_init_list = draw_camera_frustum_geometry(c2w_init.cpu().detach().cpu().numpy(), self.dataset.H, self.dataset.W,
                                                        fx_init, fx_init,
                                                        frustum_length, init_color)

        # gt poses
        frustum_gt_list = self.get_gt_poses(os.path.join('./exp/teeth_noise', 'cameras_sphere.npz'), cam_num, color=all_color[2], length=frustum_length)

        geometry_to_draw = []
        geometry_to_draw.append(frustum_est_list)
        geometry_to_draw.append(frustum_init_list)
        geometry_to_draw.append(frustum_gt_list)
        
        # mesh
        mesh = o3d.io.read_triangle_mesh(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(iter_show)))
        mesh.compute_vertex_normals()
        geometry_to_draw.append(mesh)

        o3d.visualization.draw_geometries(geometry_to_draw)

    def validate_mesh(self, world_space=False, resolution=256, threshold=0.0):        
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('-c', '--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('-d', '--img_dir', type=str, default='image')
    parser.add_argument('-psfx', '--npz_postfix', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, img_dir=args.img_dir, npz_postfix=args.npz_postfix, is_continue=args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode.startswith('validate_mesh'):
        iter_show = args.mode.split('_')[-1]
        runner.load_checkpoint(('ckpt_{:0>6d}.pth').format(int(iter_show)))
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    elif args.mode.startswith('getfeats'):
        _, iter_show = args.mode.split('_')
        runner.load_checkpoint(('ckpt_{:0>6d}.pth').format(int(iter_show)))
        runner.val_all_imgs(resolution_level=1, gen_depth_for_finetune=True, both_mask=False)
    elif args.mode.startswith('valimg'):
        _, iter_show = args.mode.split('_')
        runner.load_checkpoint(('ckpt_{:0>6d}.pth').format(int(iter_show)))
        runner.val_all_imgs(resolution_level=2, gen_depth_for_finetune=False, both_mask=True)
    elif args.mode.startswith('showcam'):
        _, iter_show = args.mode.split('_')
        runner.load_pnf_checkpoint(('pnf_{:0>6d}.pth').format(int(iter_show)))
        runner.show_cam_pose(int(iter_show))
