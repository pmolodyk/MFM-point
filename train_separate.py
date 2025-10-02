import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from datasets import get_dataset, get_dataloader
from model import get_model
from util.ema_util import ExponentialMovingAverage
from util.file_utils import name_category
import argparse
import matplotlib.pyplot as plt
import wandb
from util.visualize import *
import hashlib

def generate_run_id(exp_name):
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

def get_gamma(time_sampler):
    if time_sampler == 'uniform':
        gamma = lambda t: t
    elif time_sampler == 'sqrt':
        gamma = lambda t: torch.sqrt(t)
    elif time_sampler == 'lognormal':
        gamma = lambda t: t
    return gamma

def sample_t(time_sampler, sz, sampler_params=None):
    if time_sampler == 'lognormal':
        return nn.functional.sigmoid(sampler_params[0] + torch.randn(size=(sz,)) * sampler_params[1])
    else:
        return torch.rand(size=(sz,))

def train(opt):
    device = f'cuda:0'
    T_s = opt.stage_start_time
    T_e = opt.stage_end_time
    
    # get time_sampler
    gamma = get_gamma(opt.time_sampler)
    sampler_params = (opt.logn_m, opt.logn_s)

    # Get Data
    train_dataset, _ = get_dataset(opt.dataroot, opt.dataset_name, opt.npoints, opt.category, opt.random_subsample, opt.downsample_ratio)
    dataloader, _ = get_dataloader(opt, train_dataset, None)

    # Get Model
    model = get_model(opt)
    model = model.to(device)
    model = nn.DataParallel(model)

    # Get optimizer
    optimizer= optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))
    ema = ExponentialMovingAverage(model.parameters(), decay=opt.ema)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)
    
    # Get logger
    category_name = name_category(opt.category)

    parent_dir = os.path.join(opt.root_dir, 'train_logs', category_name)
    exp_path = os.path.join(parent_dir, opt.exp_dir)
    os.makedirs(exp_path, exist_ok=True)
    
    # Make log file
    with open(os.path.join(exp_path, 'log.txt'), 'w') as f:
        f.write("Start Training")
        f.write('\n')

    # save configurations
    jsonstr = json.dumps(opt.__dict__, indent=4)
    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        f.write(jsonstr)

    # Setup wandb
    if opt.use_wandb:
        run_name = f'{opt.exp_dir}_{opt.dataset_name}_{category_name}_{opt.niter}-inter'
        if opt.reattach_wandb:
            run_id = generate_run_id(run_name)
        else:
            run_id = None
        wandb_run = wandb.init(mode='online', project='pc_gen', name=run_name, id=run_id)

    # Load checkpoint
    start_epoch = 0
    if opt.resume and os.path.exists(os.path.join(exp_path, f"latest_checkpoint.pth")):
        checkpoint = torch.load(os.path.join(exp_path, f"latest_checkpoint.pth"))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        lr_scheduler = checkpoint['lr_scheduler']
        ema = checkpoint['ema']
        start_epoch = checkpoint['epoch'] + 1

    # Start training
    for epoch in range(start_epoch, opt.niter):
        for i, data in enumerate(dataloader):
            
            optimizer.zero_grad()

            with torch.no_grad():
                ## get the original data
                if opt.stage == 0:
                    X_L = data['train_points'][0].float().to(device).repeat_interleave(opt.downsample_ratio, dim=1)
                    X_H = data['train_points'][1].float().to(device)
                    noise = torch.randn_like(X_H).to(device)
                elif opt.stage == 1:
                    X_L = data['train_points'][0].float().to(device)
                    X_H = data['train_points'][0].float().to(device)
                    noise = torch.randn_like(X_H).to(device) / np.sqrt((opt.downsample_ratio)) * opt.gaussian_multiplier
                else:
                    raise NotImplementedError                

                X_0 = T_s * X_L + (1-T_s) * noise
                X_H = T_e * X_H + (1-T_e) * noise

                ts = gamma(sample_t(opt.time_sampler, len(X_L), sampler_params).to(device))
                xts = (1-ts)[:,None, None] * X_0 + ts[:,None, None] * X_H

            output = model([xts], [ts])[0]

            # compute loss and update
            loss = opt.loss_weights * ((output - (X_H - X_0))**2).mean()
            loss.backward()
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update(model.parameters())

            if i % opt.print_freq == 0:
                with open(os.path.join(exp_path, 'log.txt'), 'a') as f:
                    f.write(f'Epoch {epoch}: {loss.item()}')
                    f.write('\n')
                if opt.use_wandb:
                    logdata = {'epoch': epoch, 'lr': lr_scheduler.get_last_lr()[0]}
                    logdata.update({f'loss_0': loss})

                    grad_norms = list()
                    for p in model.parameters():
                        param_norm = p.grad.detach().data.norm(2)
                        grad_norms.append(param_norm.item() ** 2)
                    logdata.update({f'max_grad_norm': max(grad_norms), f'mean_grad_norm': sum(grad_norms) / len(grad_norms)})

                    wandb_run.log(logdata)

        lr_scheduler.step()
        
        # save whole ckpts to latest_checkpoint.pth
        state = {'optimizer': optimizer, 
                 'model': model, 
                 'ema': ema, 
                 'lr_scheduler': lr_scheduler, 
                 'epoch': epoch}
        
        torch.save(state, os.path.join(exp_path, 'latest_checkpoint.pth'))

        # if True:
        if (epoch+1) % opt.saveIter == 0:
            torch.save(state, os.path.join(exp_path, f'checkpoint_{epoch+1}.pth'))

        # if True:
        if (epoch+1) % opt.vizIter == 0:
            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            with torch.no_grad():
                ts = torch.linspace(0, 1, opt.time_num)
                ts = ts[:,None].repeat(1, len(noise)).to(noise.device)
                x = X_0
                for t, t_plus in zip(ts[:-1], ts[1:]):
                    x = x + model([x], [gamma(t)])[0] * ((gamma(t_plus) - gamma(t))[:,None,None])
                
                generated = x.cpu().numpy()

            if opt.use_wandb:
                visualize_pointcloud_batch(os.path.join(exp_path, f'generated_{epoch}.png'), torch.from_numpy(generated), None, None, None, wandb_run=wandb_run) 
            else:
                visualize_pointcloud_batch(os.path.join(exp_path, f'generated_{epoch}.png'), torch.from_numpy(generated), None, None, None)
            ema.restore(model.parameters())
            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--category', nargs='+', type=str, default='airplane')
    parser.add_argument("--exp_dir", type=str, default="temp")
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--stage', type=int)
    parser.add_argument('--stage_start_time', type=float, default=0)
    parser.add_argument('--stage_end_time', type=float, default=1.0)
    parser.add_argument('--npoints', type=int, default=2048)
    parser.add_argument('--downsample_ratio', type=int, default=4)
    parser.add_argument('--time_sampler', type=str, default='sqrt', choices=['uniform', 'sqrt', 'lognormal'])
    parser.add_argument('--logn_m', type=float, default=None, help='location parameter used only for lognormal sampling')
    parser.add_argument('--logn_s', type=float, default=None, help='scale parameter used only for lognormal sampling')
    
    # Dataset / Dataloader
    parser.add_argument('--dataroot', default='../../data/shapenet')
    parser.add_argument('--dataset_name', default='preprocessed_shapenet', choices=['shapenet',
                                                                                    'modelnet',
                                                                                    'preprocessed_shapenet',
                                                                                    'preprocessed_modelnet'])
    parser.add_argument('--random_subsample', action='store_false', default=True)
    parser.add_argument('--bs', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--nc', default=3)

    #Model Params
    parser.add_argument('--model_name', default='pvcnn', choices=['pvcnn', 'dit', 'transformer', 'pvcnn_split'], help="name of the model")
    parser.add_argument('--voxel_res', nargs='+', type=float, default=None, help='voxel resolution for pvcnn experiments')
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    ## transfomer-based config
    parser.add_argument('--max_point_num', type=int, default=2048)
    parser.add_argument('--base_point_num', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--embedding_intermediate_size', type=int, default=64)

    # Training Configurations
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')
    parser.add_argument('--lr_scheduler_min', type=float, default=1e-5, help='min lr for cosine scheduler')
    parser.add_argument('--loss_weights', type=float, default=1, help='loss weights for each stage')
    parser.add_argument('--ema', type=float, default=0.9999)
    parser.add_argument('--gpu', nargs='+', type=str, default='0', help='GPU id to use. None means using all available GPUs.')
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--gaussian_multiplier', type=float, default=1.0)
    
    '''eval'''
    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')
    parser.add_argument('--saveIter', type=int, default=100, help='unit: epoch')
    parser.add_argument('--vizIter', type=int, default=10, help='unit: epoch')
    parser.add_argument('--print_freq', type=int, default=100, help='unit: iter')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--reattach_wandb', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--time_num', type=int, default=100)

    opt = parser.parse_args()
    train(opt)