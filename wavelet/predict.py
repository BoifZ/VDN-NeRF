import scipy
from utils import evaluate
import argparse
from model import Model, InfModel
from load_save_utils import *
import numpy as np
import os
import scipy.io as io
from imageio import imread, imsave
from data import getNoTransform
from PIL import Image
import cv2 as cv
from data import depthDatasetMemory
# import open3d as o3d

import numpy as np
import torch
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from PIL import Image
from io import BytesIO
import random
import json
from tqdm import tqdm

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError(
            'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))

        return img.float().div(255)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(
            torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)

    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def feat_to_img(featmap, max_value=None, colormap='bone'):
    B, H, W = featmap.shape
    # print(featmap.shape)
    featvecs = featmap.reshape(B, -1)  # np.reshape(featmap, (B, -1))
    cov_mat = np.cov(featvecs)
    a, vect = np.linalg.eig(cov_mat)
    # print(vect.shape)
    feat_transed = np.dot(np.linalg.inv(vect)[:3], featvecs)
    # print(feat_transed.shape)

    featmap = np.reshape(feat_transed, (3, H, W))
    # print(featmap.shape)
    if max_value is None:
        max_value = featmap[featmap < np.inf].max().item()
        min_value = featmap[featmap != np.nan].min().item()
        feat_rgb = (featmap-min_value)/(max_value-min_value)
        feat_rgb = 0.5 + feat_rgb*0.5
        print(max_value)
        print(min_value)
    else:
        feat_rgb = (featmap/max_value).clip(-1,1)*0.5+0.5

    # norm_array[norm_array == np.inf] = np.nan
    # array = COLORMAPS[colormap](norm_array).astype(np.float32)
    feat_rgb = np.transpose(feat_rgb, (1,2,0))
    return feat_rgb


# Arguments
parser = argparse.ArgumentParser(description='Single Image Depth Prediction with Wavelet Decomposition')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--logdir', type=str, default='log')
parser.add_argument('--model_name', type=str, default='DenseNetWaveLet')
parser.add_argument('--disparity', action="store_true")
parser.add_argument("--loss_scales",
                    nargs="+",
                    type=int,
                    help="scales at which outputs are computed",
                    default=[0, 1, 2, 3])
parser.add_argument("--output_scales",
                    nargs="+",
                    type=int,
                    help="scales used in the loss",
                    default=[0, 1, 2, 3])
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--num_workers', default=0, type=int, help='batch size')
parser.add_argument('--log_histogram', action="store_true")
parser.add_argument('-ckpt', '--ckpt_folder', type=str, default='/home/zbf/Desktop/DeuS/wavelet/log/DenseNetWaveLet/teeth_train/models/weights_20')  # continues to train
parser.add_argument('--ckpt_name', type=str, default='model.pth')  # continues to train
parser.add_argument('--normalize_input', action="store_true")
parser.add_argument('--supervise_LL', action="store_true", default=True)
parser.add_argument('--encoder_type', type=str, choices=["densenet", "mobilenet"], default="densenet")
# parser.add_argument('--encoder_type', type=str, choices=["densenet"], default="densenet")
parser.add_argument('--use_wavelets', action="store_true", default=True)
parser.add_argument('--no_pretrained', action="store_true", default=False)
parser.add_argument('--dw_waveconv', action="store_true")
parser.add_argument('--dw_upconv', action="store_true")
parser.add_argument('-full', '--is_full', action="store_true")
parser.add_argument('--use_224', action="store_true", default=False)
parser.add_argument('-d','--pic_routine', default='./predict_data/')
parser.add_argument('--type', type=str, default='')  # continues to train

args = parser.parse_args()

torch.cuda.set_device(args.gpu)
logpath = os.path.join(args.logdir, args.model_name)
args.check_point_folder = os.path.join(logpath, 'checkpoint')

if args.no_pretrained:
    args.pretrained_encoder = False
else:
    args.pretrained_encoder = True

print("Creating model...", end="")
model = Model(args).cuda()
# model = InfModel(args).cuda()
print(' ...Model created.')

model = load_model(model, os.path.join(args.ckpt_folder, args.ckpt_name))

print(' ...Model loaded.')

type = args.type
# is_full = True
depth_folder = os.path.join(args.pic_routine, 'wavelet_feats')
if type == 'msk':
    print('[Debug] using mask from file')
    depth_folder += '_msk'

if args.is_full: depth_folder +='_full'
os.makedirs(depth_folder, exist_ok=True)
model.eval()
file_list = os.listdir(args.pic_routine)
if os.path.exists('./output') is False:
    os.mkdir('./output')
print('[Info] find {} images in folder {}'.format(len(file_list), args.pic_routine))


for i in file_list:
    if not i.endswith('.png'): continue
    file_name = i[:-4]
    pic_file = os.path.join(args.pic_routine,i)
    
    pic_data = cv.imread(pic_file, -1)
    if type == 'msk':
        msk_file = os.path.join(args.pic_routine,'mask',i)
        # msk_file = os.path.join('/home/zbf/Desktop/DeuS/data/bmvs/sclup/mask', i)
        mask = cv.imread(msk_file)/255
        pic_data = pic_data*mask+(1-mask)*255
    if args.is_full:
        pic_data = cv.resize(pic_data, (0,0), fx=2, fy=2)
    if pic_data.shape[-1]==4:
        print('[Debug] rgba to rgb')
        pic_data, a = pic_data[:,:,:3], pic_data[:,:,3:]
        a = a/255.0
        pic_data = pic_data*a + (1.0-a)*255
    pic_data = to_tensor(pic_data).cuda()
    pic_data = pic_data.unsqueeze(0)

    with torch.no_grad():
        enc_feats = [feat.detach().cpu() for feat in model.encoder(pic_data)]
    for idx, res in enumerate(enc_feats):
        # print(idx, res.shape)
        if idx != 0: continue
        # if idx != 2: continue
        # print(res.shape)
        image = np.uint8(feat_to_img(res.squeeze().numpy(), max_value=8)*255)
        # image.show()
        # imsave(os.path.join(depth_folder, '{}_{}.png').format(i[:-4], idx), image)
        depth_feat = res.numpy()
        os.makedirs(os.path.join(depth_folder, str(idx)), exist_ok=True)
        np.save(os.path.join(depth_folder, str(idx), i[:-4]+'.npy'), depth_feat)
        print('{} Saved'.format(i))
