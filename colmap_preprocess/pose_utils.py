import numpy as np
import os
import shutil
import sys
import imageio
import skimage.transform
import trimesh

from colmap_wrapper import run_colmap
import colmap_read_model as read_model


def load_colmap_data(realdir):
    print('[Info]loading colmap data from folder ', os.path.join(realdir, 'sparse/0'))
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    print(h, w, f)
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    # print(len(imdata))
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    # exists_img = {}
    names = [imdata[k].name for k in imdata]
    perm = np.argsort(names)
    print( '[Info] Images #', len(names))
    # nn = [int(n[2:-4]) for n in names]
    # perm = np.argsort(nn)
    for cnt, k in enumerate(imdata):
        # exists_img[imdata[k].id] = cnt
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    # print('[Info] exists_img', exists_img)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    # print('[debug] c2w_mats', c2w_mats[0])
    # print('[debug] w2c_mats', w2c_mats[0])
    return poses, pts3d, perm, np.array(names, dtype=str)


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    # print(poses.shape[-1])
    for k in pts3d:
        flag = False
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) <= ind - 1: 
                print(pts3d[k].image_ids)
                continue
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            flag = True
            cams[ind-1] = 1
        if flag:
            pts_arr.append(pts3d[k].xyz)
            vis_arr.append(cams)

    pts = np.stack(pts_arr, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(basedir, 'sparse_points.ply'))

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape )

    poses = np.moveaxis(poses, -1, 0)
    poses = poses[perm]
    np.save(os.path.join(basedir, 'poses.npy'), poses)


def minify_v0(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    def downsample(imgs, f):
        sh = list(imgs.shape)
        sh = sh[:-3] + [sh[-3]//f, f, sh[-2]//f, f, sh[-1]]
        imgs = np.reshape(imgs, sh)
        imgs = np.mean(imgs, (-2, -4))
        return imgs
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgs = np.stack([imageio.imread(img)/255. for img in imgs], 0)
    
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)
        
        if isinstance(r, int):
            imgs_down = downsample(imgs, r)
        else:
            imgs_down = skimage.transform.resize(imgs, [imgs.shape[0], r[0], r[1], imgs.shape[-1]],
                                                order=1, mode='constant', cval=0, clip=True, preserve_range=False, 
                                                 anti_aliasing=True, anti_aliasing_sigma=None)
        
        os.makedirs(imgdir)
        for i in range(imgs_down.shape[0]):
            imageio.imwrite(os.path.join(imgdir, 'image{:03d}.png'.format(i)), (255*imgs_down[i]).astype(np.uint8))
            

def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100./r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    # imgs = [imageio.imread(f, ignoregamma=True)[...,:3]/255. for f in imgfiles]
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

def move_images(basedir, names):
    import cv2 as cv
    np.savetxt(os.path.join(basedir,'images_name.txt'), names, fmt='%s')
    for i, n in enumerate(names):
        origin_path = os.path.join(basedir, '', n)
        # new_dir = os.path.join(basedir,'preprocessed/image')
        new_dir = os.path.join(basedir,'n_image')
        os.makedirs(new_dir, exist_ok=True)
        # os.makedirs(os.path.join(basedir, 'mask'), exist_ok=True)
        shutil.copyfile(origin_path, os.path.join(new_dir, '{:0>3d}.png'.format(i)))
        # origin_mask = os.path.join(basedir, 'mask', n)
        # new_msk = os.path.join(basedir,'n_mask')
        # os.makedirs(new_msk, exist_ok=True)
        # shutil.copyfile(origin_mask, os.path.join(new_msk, '{:0>3d}.png'.format(i)))
        # img = cv.imread(origin_path)
        # if img.shape[0]<img.shape[1]:
        #     img = cv.resize(img, (1200, 900))
        # else:
        #     img = cv.resize(img, (900, 1200))
        #     img = np.transpose(img, (1,0,2))
        #     # print(img.shape)
        # cv.imwrite(os.path.join(new_dir, '{}.png'.format(n[:-4])), img)
        # cv.imwrite(os.path.join(basedir, 'mask', '{}.png'.format(n[:-4])), np.ones_like(img) * 255)
            
    
def gen_poses(basedir, match_type, factors=None):
    print('[Info] base_dir:', basedir)
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        print('exists')
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print( 'Need to run COLMAP' )
        exit(111)
        run_colmap(basedir, match_type)
    else:
        print('Don\'t need to run COLMAP')
        
    print('Post-colmap')
    
    poses, pts3d, perm, names = load_colmap_data(basedir)
    names = names[perm]
    # move_images(basedir, names)

    save_poses(basedir, poses, pts3d, perm)
    
    if factors is not None:
        print( 'Factors:', factors)
        minify(basedir, factors)
    
    print( 'Done with imgs2poses' )
    
    return True
    
