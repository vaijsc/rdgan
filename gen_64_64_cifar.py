# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import torch
import numpy as np

import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.lmdb_datasets import LMDBDataset


from torch.multiprocessing import Process
import torch.distributed as dist
import shutil

from util.data_process import getCleanData, getMixedData
from util.args_parser import args_parser
from util.utility import copy_source, broadcast_params, q_sample_pairs, sample_posterior, sample_from_model, select_phi
from util.diffusion_coefficients import get_time_schedule, get_sigma_schedule


class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

#%%
def train(rank, gpu, args):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large, Discriminator_64
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    from EMA import EMA
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    
    nz = args.nz #latent dimension
    
    # dataset = getCleanData(args.dataset)
    # if args.perturb_dataset == 'none':
    #     dataset = getCleanData(args.dataset, image_size=args.image_size)
    # else:
    #     dataset = getMixedData(args.dataset, args.perturb_dataset, percentage = args.perturb_percent, image_size=args.image_size, shuffle=args.shuffle)
      
    # print("Finish loading dataset")

    # data_loader = torch.utils.data.DataLoader(dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=False,
    #                                            num_workers=4,
    #                                            pin_memory=True,
    #                                            drop_last = True)
    
    netG = NCSNpp(args).to(device)
    
    broadcast_params(netG.parameters())
    
    exp = args.exp
    # parent_dir = f"./saved_info/{args.version}/dd_gan/{args.dataset}"
    
    algo = 'rdgan'
    if args.phi1 == 'none':
        algo = 'ddgan'
    parent_dir = f'{algo}/{args.dataset}'
    
    if args.perturb_percent > 0:
        # real_img_dir += f'_{int(args.perturb_percent)}p_{args.perturb_dataset}'
        parent_dir += f'_{int(args.perturb_percent)}p_{args.perturb_dataset}'
    
    # real_img_dir += f'/{args.version}/stat_{args.epoch_id}.npy'
    # parent_folder = parent_dir + f'/{args.version}/'
    parent_dir += f'/{args.version}'
    parent_dir = f'./saved_info/' + parent_dir
    save_dir = f'./test_images/' + parent_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    exp_path = parent_dir
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    
    ckpt = torch.load(f'{parent_dir}/netG_{args.epoch}.pth', map_location=device)
    
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()
    
    real_data = None
    # for iteration, (x, y) in enumerate(tqdm(data_loader)):
    #     real_data = x.to(device, non_blocking=True)
    #     break
    x_t_1 = torch.randn(64, 3, args.image_size, args.image_size).to(device)
    fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
    torchvision.utils.save_image(fake_sample, os.path.join(save_dir, f'sample_discrete_epoch_{args.epoch}_{args.seed}.png'), normalize=True)
            


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()    
#%%
if __name__ == '__main__':
    args = args_parser()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:        
        init_processes(0, size, train, args)
   
                