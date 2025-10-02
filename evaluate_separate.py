import os
import json
import numpy as np
import torch
from ml_collections import config_dict
from datasets import get_dataset, get_dataloader
from model import get_model
from util.visualize import *
from util.file_utils import *
from tqdm import tqdm
import argparse
from train_separate import get_gamma
from einops import rearrange
import math

def sample_block_noise(B, N, dnsample=4):
    '''
    Input: B, N (batch_size, number of points)
    Output: Block noise (B x N x 3)
    '''
    assert N % dnsample == 0
    N_block = N // dnsample
    gamma = 1 / (dnsample - 1)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dnsample), 
                torch.eye(dnsample) * (1 + gamma) - torch.ones(dnsample, dnsample) * gamma)
    block_number = B * N_block * 3
    noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, self.dnsample]
    noise = noise.reshape(B, N_block, 3, dnsample)
    noise = rearrange(noise, 'b n z d -> b (n d) z')
    return noise


parser = argparse.ArgumentParser()
# Dataset / Dataloader
parser.add_argument('--exp_names', nargs='+', type=str, default=['pvcnn_2048_base_low', 'pvcnn_2048_base_high_T0.6'])
parser.add_argument('--category', nargs='+', type=str, default='airplane')
parser.add_argument('--npoints', type=int, default=2048)
parser.add_argument('--time_sampler', type=str, default='sqrt', choices=['uniform', 'sqrt'])
parser.add_argument('--time_num_low', type=int, default=1000)
parser.add_argument('--time_num_high', type=int, default=400)
parser.add_argument('--downsample_ratio', type=int, default=4)
parser.add_argument('--num_epoch_low', type=int, default=None)
parser.add_argument('--num_epoch_high', type=int, default=None)
parser.add_argument('--random_subsample', action='store_false', default=True)
parser.add_argument('--shuffle_test', action='store_true', default=False)
parser.add_argument('--take_items', type=int, default=None)
parser.add_argument('--bs', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='workers')
parser.add_argument('--nc', default=3)
parser.add_argument('--dataroot', default='../../data/ShapeNetCore.v2.PC15k')
parser.add_argument('--dataset_name', default='shapenet', choices=['shapenet'])
cfg = parser.parse_args()

exp_names = cfg.exp_names
device='cuda:0'

### logger
exp_id = os.path.splitext(os.path.basename(__file__))[0]
dir_id = os.path.dirname(__file__)
output_dir = get_output_dir(dir_id, exp_id)
copy_source(__file__, output_dir)
logger = setup_logging(output_dir)
ctgr = cfg.category

# get timesampler
gamma = get_gamma(cfg.time_sampler)


### TS, TE, MODELS
TS = []
TE = []
MODELS = []

category_name = name_category(cfg.category)
gaussian_multipliers = [1.0, 1.0]

for i, exp_name in enumerate(exp_names):
    model_path = os.path.join('train_logs', category_name, exp_name)
    logger.info(model_path)

    # get config
    cfg_dict = json.load(open(os.path.join(model_path, 'config.json')))
    opt = config_dict.ConfigDict()
    for key, item in cfg_dict.items():
        if key == 'category' and isinstance(item, str):
            item = [item]
        setattr(opt, key, item)
    
    if not hasattr(opt, 'gaussian_multiplier'):
        opt.gaussian_multiplier = 1.0

    gaussian_multipliers[1 - opt.stage] = opt.gaussian_multiplier

    # get model
    if cfg.num_epoch_low is None:
        state_dict = torch.load(os.path.join(model_path, f'latest_checkpoint.pth'))
    else:
        if i == 0:
            state_dict = torch.load(os.path.join(model_path, f'checkpoint_{cfg.num_epoch_low}.pth'))
        else:
            state_dict = torch.load(os.path.join(model_path, f'checkpoint_{cfg.num_epoch_high}.pth'))
    ema = state_dict['ema']
    model = state_dict['model']
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()
    MODELS.append(model)
    TS.append(opt.stage_start_time)
    if hasattr(opt, 'stage_end_time'):
        TE.append(opt.stage_end_time)
    else:
        TE.append(1.0)

    print(f'Restored model from epoch {cfg.num_epoch_low} lowres and {cfg.num_epoch_high} highres with start time {opt.stage_start_time}')
    print(f'start times: {TS}')
    print(f'end times: {TE}')
    print(f'gaussian_multipliers: {gaussian_multipliers}')

# assert TS[1] == TE[0], "Different times not supported yet"
assert TS[0] == 0, "Lowest resolution must start at 0"
assert TE[-1] == 1, "Highest resolution must end at 1"

if isinstance(opt.category, str):
    opt.category = [opt.category]
opt.category = ctgr
# get test dataset
_, test_dataset = get_dataset(cfg.dataroot, 'shapenet', opt.npoints, opt.category, True, 4)
_, test_dataloader = get_dataloader(opt, None, test_dataset, shuffle_test=cfg.shuffle_test)


samples = []
ref = []
total_items = 0
for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):
    
    real = data['test_points']
    
    # generate low
    with torch.no_grad():
        model = MODELS[0]
        noise = torch.randn((len(real), opt.npoints//opt.downsample_ratio, 3)).to(device) / np.sqrt(opt.downsample_ratio) * gaussian_multipliers[0]
        noise = noise.to(device)
        ts = torch.linspace(0, 1, cfg.time_num_low)
        print(f'Timesteps used for low resolution: {len(ts)}')
        ts = ts[:,None].repeat(1, len(noise)).to(noise.device)
        x = noise
        for t, t_plus in zip(ts[:-1], ts[1:]):
            x = x + model([x], [gamma(t)])[0] * ((gamma(t_plus) - gamma(t))[:,None,None])
        generated = x

    # generate high
    with torch.no_grad():
        for k, (model, T) in enumerate(zip(MODELS[1:], TS[1:])):
            ts = torch.linspace(0, 1, int(cfg.time_num_high))
            print(f'Timesteps used for high resolution: {len(ts)}')
            ts = ts[:,None].repeat(1, len(noise)).to(noise.device)
            x = generated.repeat_interleave(opt.downsample_ratio, dim=1)
            noise = torch.randn_like(x)
            if TE[0] != 1:
                alpha = (1 - TS[1]) / math.sqrt(opt.downsample_ratio / (opt.downsample_ratio - 1))
                B, N, _ = x.shape
                x = x + alpha * sample_block_noise(B, N, dnsample=opt.downsample_ratio).to(x.device)
            else:
                x = T * x + (1-T) * noise
            for t, t_plus in zip(ts[:-1], ts[1:]):
                x = x + model([x], [gamma(t)])[0] * ((gamma(t_plus) - gamma(t))[:,None,None])

            generated = x.cpu()
    
    # postprocess
    m, s = data['mean'].float(), data['std'].float()
    visualize_pointcloud_batch('generated.png', torch.from_numpy(generated.cpu().numpy()), None, None, None)    
    samples.append(s*generated+m)
    ref.append(s*real+m)
    total_items += real.shape[0]
    if cfg.take_items is not None and cfg.take_items < total_items:
        break

if cfg.take_items is not None:
    samples = samples[:cfg.take_items]
    ref = ref[:cfg.take_items]
samples = torch.cat(samples, dim=0)
ref = torch.cat(ref, dim=0)

print('Total shapes:', samples.shape, ref.shape)

def evaluate_gen(opt, ref_pcs, sample_pcs, logger):
    from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
    from metrics.evaluation_metrics import compute_all_metrics

    logger.info("Generation sample size:%s reference size: %s"  % (sample_pcs.size(), ref_pcs.size()))


    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, cfg.bs)
    results = {k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in results.items()}

    print(results)
    logger.info(results)

    jsd = JSD(sample_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    print('JSD: {}'.format(jsd))
    logger.info('JSD: {}'.format(jsd))

evaluate_gen(opt, ref, samples, logger)