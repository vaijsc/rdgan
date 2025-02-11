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
from torch.optim import Adam
import shutil
from util.data_process import get_datasets
from util.args_parser import args_parser
from util.utility import copy_source, broadcast_params, select_phi
from util.diffusion_coefficients import get_time_schedule, get_sigma_schedule
from torch.utils.data import DataLoader
from score_sde.models.toy_normal_distribution import Toy_Generator, Toy_Discriminator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

import torch
import os
import shutil
import argparse
import torch.nn.functional as F
from util.diffusion_coefficients import extract
import torch.optim as optim
from util.diffusion_coefficients import get_time_schedule, get_sigma_schedule
import ipdb
import warnings
warnings.filterwarnings("ignore")

def sample_from_model(coefficients, generator, n_time, x_init, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new, noise = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
    return x

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


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one, noise

def sample_posterior(coefficients, x_0, x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        
        
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        
        return mean + nonzero_mask[:,None] * torch.exp(0.5 * log_var) * noise, noise
    
    sample_x_pos, noise = p_sample(x_0, x_t, t)
    
    return sample_x_pos, noise

def draw_samples_from_models(coefficients, generator, n_time, x_init, opt, tar_dataloader, coeff, device, savepath):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new, noise = sample_posterior(coefficients, x_0, x, t)
            x_old = x
            x = x_new.detach()
            
            for iteration, real_data in enumerate(tar_dataloader):
                real_data = real_data.to(device)
                tar_distribution = q_sample(coeff, real_data, torch.full((args.num_data,), i).to(device))
            sources = []
            preds = []
            sources.append(x_old.detach().cpu().numpy())
            preds.append(x.detach().cpu().numpy())

            sources = np.concatenate(sources)
            preds = np.concatenate(preds)
            # ipdb.set_trace()
            df = pd.DataFrame({'target density': tar_distribution.cpu().numpy()[:,0], 'generated density': preds[:,0]})

            # joint distribution scatter plot
            plt.scatter(sources[:,0], preds[:,0])
            plt.xlabel('Source', fontsize=20)  # Set the X-axis label font size
            plt.ylabel('Target', fontsize=20)  # Set the Y-axis label font size
            plt.xticks(fontsize=14)  # Set the X-axis tick label font size
            plt.yticks(fontsize=14)  # Set the Y-axis tick label font size
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, f"Transport map at time step {i}.png"))
            plt.close()

            # # target density plot
            # np.save(os.path.join(args.savepath, f'{args.exp}_{args.phi1}_{args.phi2}_num{args.num_data}_out{args.p}_tau{tau}_lmbda{lmbda}_epoch{epoch+1}.npy'), 
            #                      {'source_density': x_src.cpu().numpy()[:,0], 'target density': tar_dataset.dataset.detach().cpu().numpy()[:,0], 'generated density': preds[:,0]})
            # sns.set(style="whitegrid")
            sns_plot = sns.kdeplot(df, fill=True, y=None)
            sns.move_legend(sns_plot, "upper left")
            # sns_plot.legend(fontsize=12)
            fig = sns_plot.get_figure()
            # plt.legend(loc="upper left")
            if i == 0:
                plt.xlim(-2,2)
            # plt.ylim(0,2.5)
            # plt.xticks([])
            # plt.yticks([]) 
            # plt.legend(loc="upper left", bbox_to_anchor=(0, 1))
            ax = plt.gca()
            plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text
            plt.setp(ax.get_legend().get_title(), fontsize='20') # for legend title
            ax.set_xlabel("x", fontsize=20)  # Set the X-axis label font size
            ax.set_ylabel("Density", fontsize=20)  # Set the Y-axis label font size
            ax.tick_params(axis='both', which='major', labelsize=14)  # Set the tick label font size
            # ax.get_xaxis().set_visible(True)
            # ax.get_yaxis().set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_yscale("symlog", linthresh=0.02)
            ax.set_yticks([0.01, 0.1, 0.2, 0.5, 1, 2])
            ax.set_yticklabels([0.01, 0.1, 0.2, 0.5, 1, 2])
            plt.tight_layout()
            fig.savefig(os.path.join(savepath, f"Generation distribution at time step {i}.png"))
            plt.close()
            print(f'Finish step {i}!')
    return x

def main(args):
    device='cuda:0'
    batch_size = args.batch_size
    command_line = ' '.join(sys.argv)

    nz = args.nz #latent dimension
    # dataloader
    src_dataset, tar_dataset = get_datasets(args)
    src_dataloader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    tar_dataloader = DataLoader(tar_dataset, batch_size=args.num_data, shuffle=True, drop_last=True)

    

    print(args.phi1)
    algo = 'rdgan'
    if args.phi1 == 'none':
        algo = 'ddgan'
    else:
        phi1 = select_phi(args.phi1)
        phi2 = select_phi(args.phi2)
    
    # ipdb.set_trace()
    savepath = f'./train_logs/outlier/{algo}/{args.target_name}_{int(args.p*100)}p/{args.version}'

    # model
    netG = Toy_Generator(args).to(device)
    checkpoint = torch.load(f'{savepath}/netG_{args.epoch}.pth')
    netG.load_state_dict(checkpoint)
    netG.eval()


    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    global_step = 0

    

    with torch.no_grad():
        x_t_1 = torch.randn((args.num_data, 1), device=device)
        draw_samples_from_models(pos_coeff, netG, args.num_timesteps, x_t_1, args, tar_dataloader, coeff, device, savepath)

        # sources = []
        # preds = []
        
        # # print(type(tar_dataset.dataset.detach()))
        # # sys.exit()
        # for iteration, x in enumerate(tar_dataloader):
        #     x = x.to(device)
        #     tar_distribution = q_sample(coeff, x, torch.full((args.num_data,), 0).to(device))
        #     # sys.exit()
        # x_t_1 = torch.randn((args.num_data, 1), device=device)
        # x_pred = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args)
        # # print(x_pred.shape)
        # sources.append(x_t_1.detach().cpu().numpy())
        # preds.append(x_pred.detach().cpu().numpy())

        # sources = np.concatenate(sources)
        # preds = np.concatenate(preds)
        # # ipdb.set_trace()
        # df = pd.DataFrame({'target density': tar_distribution.cpu().numpy()[:,0], 'generated density': preds[:,0]})

        # # joint distribution scatter plot
        # plt.scatter(sources[:,0], preds[:,0])
        # plt.xlabel('source')
        # plt.ylabel('target')
        # plt.savefig(os.path.join(savepath, f"test1.png"))
        # plt.close()

        # # # target density plot
        # # np.save(os.path.join(args.savepath, f'{args.exp}_{args.phi1}_{args.phi2}_num{args.num_data}_out{args.p}_tau{tau}_lmbda{lmbda}_epoch{epoch+1}.npy'), 
        # #                      {'source_density': x_src.cpu().numpy()[:,0], 'target density': tar_dataset.dataset.detach().cpu().numpy()[:,0], 'generated density': preds[:,0]})
        # # sns.set(style="whitegrid")
        # sns_plot = sns.kdeplot(df, fill=True, y=None)
        # sns.move_legend(sns_plot, "upper left")
        # fig = sns_plot.get_figure()
        # # plt.legend(loc="upper left")
        # plt.xlim(-2,2)
        # # plt.ylim(0,2.5)
        # plt.xticks([])
        # plt.yticks([]) 
        # # plt.legend(loc="upper left", bbox_to_anchor=(0, 1))
        # ax = plt.gca()
        # ax.get_xaxis().set_visible(True)
        # ax.get_yaxis().set_visible(True)
        # ax.set_yscale("symlog", linthresh=1)
        # plt.tight_layout()
        # fig.savefig(os.path.join(savepath, f"test3.png"))
        
        # plt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser('toy parameters')
    
    # Experiment description
    parser.add_argument('--exp', type=str, choices=['gaussian', 'outlier', '1d-gaussian-mixture'], help='experiment name')
    parser.add_argument('--num_data', type=int, default=4000, help='Number of data points')
    parser.add_argument('--data_dim', type=int, default=1, help='The dimensiion of data')
    parser.add_argument('--source_name', type=str, required=True, choices=['gaussian', 'p', 'q', 'outlier', 'noise'], help='Name of source dataset')
    parser.add_argument('--target_name', type=str, required=True, choices=['gaussian', 'p', 'q', 'outlier', 'noise'], help='Name of target dataset')    
    parser.add_argument('--p', type=float, default=0., help='Only for outlier test. Fraction of outlier')
    
    # settings
    # Loss configurations
    parser.add_argument('--phi1', type=str, default='none', choices=['none', 'linear', 'kl', 'softplus', 'chi'])
    parser.add_argument('--phi2', type=str, default='none', choices=['none', 'linear', 'kl', 'softplus', 'chi'])
    parser.add_argument('--tau', type=float, help='scalar value multiplied to quadratic cost functional')

    # training configurations
    parser.add_argument('--num_epoch', type=int, default=1000, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    # kl-div
    parser.add_argument('--kl_estimator', type=str, default='scipy', help='kl divergence estimator')

    parser.add_argument('--save_every', type=int, default=100, help='Evaluation every {save_every} epoch')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--beta_min', type=float, default= 0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--version', type=str, help='experiment name')
    parser.add_argument('--time_embed_dim', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=500)

    args = parser.parse_args()
    main(args)
