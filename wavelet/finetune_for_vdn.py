import time
import argparse
import datetime

import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import Model
from data import getTrainingTestingData, getNeusTrainTestData
from utils import AverageMeter, DepthNorm
from load_save_utils import *
from pytorch_wavelets import DWT

import os
import sys
import numpy as np
from train import val, log
from tqdm import tqdm


def parse_argument():
    parser = argparse.ArgumentParser(description='Single Image Depth Prediction with Wavelet Decomposition')
    parser.add_argument('-r', '--dataset_root', type=str, default='../depth_data/')
    parser.add_argument('-d', '--imgdir', type=str, default='image')
    parser.add_argument('-max', '--dpt_max', type=float, default=4)
    parser.add_argument('--case', type=str, default='lego')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('-lr', '--learning-rate', default=0.00001, type=float, help='initial learning rate')
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
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-bs', '--batch-size', default=4, type=int, help='batch size')
    parser.add_argument('--save_freq', default=30, type=int, help='checkpoint saving frequency')
    parser.add_argument('--num_workers', default=0, type=int, help='batch size')
    parser.add_argument('-ckpt', '--pretrained-ckpt', type=str, default=None)
    parser.add_argument('-c', '--continue-train', action="store_true")
    parser.add_argument('--log_histogram', action="store_true")
    parser.add_argument('--load_checkpoint', action="store_true") # continues to train
    parser.add_argument('--normalize_input', action="store_true")
    parser.add_argument('--supervise_LL', action="store_true", default=True)
    parser.add_argument('--encoder_type', type=str, choices=["densenet"], default="densenet")
    parser.add_argument('--use_wavelets', action="store_true", default=True)
    parser.add_argument('--no_pretrained', action="store_true", default=False)
    parser.add_argument('--dw_waveconv', action="store_true")
    parser.add_argument('--dw_upconv', action="store_true")
    parser.add_argument('--use_224', action="store_true", default=False)

    args = parser.parse_args()
    return args


def finetune():
    # Arguments
    args = parse_argument()
    torch.cuda.set_device(args.gpu)
    # Logging
    logpath = os.path.join(args.logdir, args.model_name, datetime.datetime.now().strftime("%m%d_%H%M")
                            + '-msk_{}_{}'.format(args.case, args.imgdir.split('image')[-1]))
    args.check_point_folder = os.path.join(logpath, 'checkpoint')

    os.makedirs(logpath, exist_ok=True)
    save_opts(logpath, args)
    with open(os.path.join(logpath, 'commandline_args.txt'), 'w') as f:
        f.write(' '.join(sys.argv[1:]))

    # Create model
    if args.no_pretrained:
        args.pretrained_encoder = False
    else:
        args.pretrained_encoder = True

    print("Creating model...", end="")
    model = Model(args).cuda()
    print(' ...Model created.')

    if args.pretrained_ckpt is not None:
        load_model_from_folder(model, args.pretrained_ckpt)

    # Training parameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    learning_rate_alpha = 0.05
    batch_size = args.batch_size

    param_to_train = model.encoder.parameters()
    for param in model.decoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(param_to_train, lr=learning_rate)
    
    def update_learning_rate(iter_step, warmup_end=0):
        if iter_step < warmup_end:
            learning_factor = iter_step / warmup_end
        else:
            alpha = learning_rate_alpha
            progress = (iter_step - warmup_end) / (epochs - warmup_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in optimizer.param_groups:
            g['lr'] = learning_rate * learning_factor

    # Load data
    root_folder = os.path.join(args.dataset_root, args.case)
    train_loader, test_loader = getNeusTrainTestData(root_folder, imgdir=args.imgdir, batch_size=batch_size,
                                                     num_workers=args.num_workers, dpt_max=args.dpt_max)
    test_iter = iter(test_loader)

    writers = {}
    for mode in ["train", "val"]:
        writers[mode] = SummaryWriter(os.path.join(logpath, mode))

    # Loss
    l1_criterion = nn.L1Loss()

    # DWT
    forward_dwt = DWT(J=4, wave='haar', mode='reflect').cuda()

    N = len(train_loader)
    start_epoch = 0
    niter = 0
    # Start training...
    for epoch in tqdm(range(start_epoch, epochs)):
        batch_time = AverageMeter()
        # losses = AverageMeter()
        losses = {}
        # Switch to train mode
        model.train()
        end = time.time()
        
        for i, sample_batched in enumerate(train_loader):
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
            mask = torch.autograd.Variable(sample_batched['mask'].cuda(non_blocking=True))

            # Normalize depth
            if args.disparity:
                depth_n = DepthNorm(depth)  # DepthNorm: maxDepth / depth
            else:
                depth_n = depth
            depth_n *= mask
            inputs = {"disp_gt": depth_n}
            inputs["image"] = image.detach().cpu()

            pad_lowestscale = nn.Identity()
            if args.use_wavelets:
                yl_gt, yh_gt = forward_dwt(depth_n)
                inputs[("wavelets", 3, "LL")] = pad_lowestscale(yl_gt)

            # Predict
            outputs = model(image)

            # Compute the loss
            total_loss = 0

            for scale in range(4):
                if args.use_wavelets:
                    inputs[("wavelets", scale, "LH")] = yh_gt[scale][:, 0, 0]
                    inputs[("wavelets", scale, "HL")] = yh_gt[scale][:, 0, 1]
                    inputs[("wavelets", scale, "HH")] = yh_gt[scale][:, 0, 2]

                if scale in args.output_scales:
                    pred = F.interpolate(outputs[("disp", scale)],
                                         scale_factor=2**scale, mode='bilinear', align_corners=True)
                    # print(pred.shape, depth_n.shape)
                    pred = pred*mask
                    l_depth = l1_criterion(pred, depth_n)

                    loss = (0.1 * l_depth)
                    if scale in args.loss_scales:
                        total_loss += loss
                    losses["loss/{}".format(scale)] = loss.item()
                    losses["loss_depth/{}".format(scale)] = l_depth.item()

            if args.use_wavelets:
                try:
                    l_LL = l1_criterion(outputs[("wavelets", 3, "LL")], yl_gt) / (2**4)
                    losses["loss_LL3"] = l_LL
                    if args.supervise_LL:
                        total_loss += l_LL
                except KeyError:
                    pass

            # Update step
            optimizer.zero_grad()
            losses["loss"] = total_loss
            total_loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter += 1
            if niter % 100 == 0:
                # Print to console
                print('Epoch: [{}][{}/{}]\t Loss {:.4f}'.format(epoch, i, N, losses["loss"]))

            if niter % 300 == 0:
                # Log to tensorboard
                test_iter = val(model, test_iter, test_loader, forward_dwt, l1_criterion, writers, niter, args, l1_criterion)

        update_learning_rate(epoch)
        # save epoch
        if epoch % args.save_freq == 0:
            save_model(model, logpath, epoch)
    save_model(model, logpath, epoch)
    print(logpath)

if __name__ == '__main__':
    finetune()
