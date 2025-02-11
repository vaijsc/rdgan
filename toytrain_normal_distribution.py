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

def main(args):
    device='cuda:0'
    batch_size = args.batch_size
    command_line = ' '.join(sys.argv)

    nz = args.nz #latent dimension
    # dataloader
    src_dataset, tar_dataset = get_datasets(args)
    src_dataloader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    tar_dataloader = DataLoader(tar_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # model
    netG = Toy_Generator(args).to(device)
    netD = Toy_Discriminator(args).to(device)

    # optimizer
    optimizerD = Adam(netD.parameters(), lr=args.lr)
    optimizerG = Adam(netG.parameters(), lr=args.lr)


    algo = 'rdgan'
    if args.phi1 == 'none':
        algo = 'ddgan'
    else:
        # phi
        phi1 = select_phi(args.phi1)
        phi2 = select_phi(args.phi2)
    # ipdb.set_trace()
    savepath = f'./train_logs/outlier/{algo}/{args.target_name}_{int(args.p*100)}p/{args.version}'
    print(savepath)
    # make savepath
    os.makedirs(savepath, exist_ok=True) 
    
    command_line = 'python3 ' + ' '.join(sys.argv)
    # Define the file name for the output file
    output_file = f"{savepath}/command_history.txt"

    # Append the command line to the output file
    with open(output_file, "w") as file:
        file.write(command_line + "\n")

    # # Print the command line to the console (optional)
    # print(f"Command saved to {output_file}: {command_line}")

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    global_step = 0

    for epoch in range(0, args.num_epoch+1): 
        print(f'Epoch {epoch}')      
        # for iteration, x in enumerate(tqdm(tar_dataloader)):
        for iteration, x in enumerate(tar_dataloader):
            for p in netD.parameters():  
                p.requires_grad = True  
        
            
            netD.zero_grad()
            
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            
            #sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1, _ = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True
            
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            
            if args.phi1 == 'none':
                errD_real = F.softplus(-D_real)
            else:
                errD_real = phi2(-D_real)
            errD_real = errD_real.mean()
            
            errD_real.backward(retain_graph=True)
            
            
            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                
                
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                    grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                
                
                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake
            latent_z = torch.randn(args.batch_size, args.nz, device=device)
            
         
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample, _ = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
                
            
            if args.phi1 == 'none':
                errD_fake = F.softplus(output)
            else:
                errD_fake = phi1(output - args.tau * torch.sum(((x_0_predict-x_tp1.detach()).view(x_tp1.detach().size(0), -1))**2, dim=1))
            errD_fake = errD_fake.mean()
            errD_fake.backward()
    
            
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            
        
            #update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            
            x_t, x_tp1, _ = q_sample_pairs(coeff, real_data, t)
            latent_z = torch.randn(args.batch_size, args.nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample, _ = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
               
            
            if args.phi1 == 'none':
                errG = F.softplus(-output)
            else:
                errG = args.tau * torch.sum(((x_0_predict-x_tp1.detach()).view(x_tp1.detach().size(0), -1))**2, dim=1) - output
            errG = errG.mean()
            
            errG.backward()
            optimizerG.step()
                
            global_step += 1
        

        # evaluation
        if epoch % args.save_every == args.save_every - 1:
            # evaluation for 1D tasks
            if args.data_dim == 1:
                with torch.no_grad():
                    sources = []
                    preds = []

                    # try: x_src = src_dataset.dataset.to(x_tar.device)
                    # except: x_src = torch.randn((args.num_data, 1), device=x_tar.device)
                    # x_src = torch.randn((args.num_data, 1), device=x_tar.device)
                    
                    x_t_1 = torch.randn((args.num_data, 1), device=device)
                    x_pred = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args)
                    # print(x_pred.shape)
                    # sources.append(x_src.detach().cpu().numpy())
                    preds.append(x_pred.detach().cpu().numpy())

                    # sources = np.concatenate(sources)
                    preds = np.concatenate(preds)
                    # ipdb.set_trace()
                    df = pd.DataFrame({'target density': tar_dataset.dataset.detach().cpu().numpy()[:,0], 'generated density': preds[:,0]})

                    # joint distribution scatter plot
                    # plt.scatter(sources[:,0], preds[:,0])
                    # plt.xlabel('source')
                    # plt.ylabel('target')
                    # plt.savefig(os.path.join(args.savepath, 
                    #                          f'joint_{args.exp}_{args.phi1}_{args.phi2}_num{args.num_data}_out{args.p}_tau{tau}_lmbda{lmbda}_epoch{epoch+1}.png'))
                    # plt.close()

                    # # target density plot
                    # np.save(os.path.join(args.savepath, f'{args.exp}_{args.phi1}_{args.phi2}_num{args.num_data}_out{args.p}_tau{tau}_lmbda{lmbda}_epoch{epoch+1}.npy'), 
                    #                      {'source_density': x_src.cpu().numpy()[:,0], 'target density': tar_dataset.dataset.detach().cpu().numpy()[:,0], 'generated density': preds[:,0]})

                    sns_plot = sns.kdeplot(df, fill=True, y=None)
                    sns.move_legend(sns_plot, "upper left")
                    fig = sns_plot.get_figure()
                    # plt.xlim(-2,2)
                    plt.ylim(0,2)
                    # plt.xticks([])
                    # plt.yticks([]) 
                    ax = plt.gca()
                    ax.get_xaxis().set_visible(True)
                    ax.get_yaxis().set_visible(True)
                    ax.set_yscale("symlog", linthresh=0.1)
                    plt.tight_layout()
                    fig.savefig(os.path.join(savepath, f"epoch{epoch+1}.png"))
                    
                    plt.close()
                    torch.save(netG.state_dict(), os.path.join(savepath, 'netG_{}.pth'.format(epoch + 1)))
                    # sys.exit()

            if args.data_dim == 2:
                pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser('toy parameters')
    
    # Experiment description
    parser.add_argument('--exp', type=str, choices=['gaussian', 'outlier', '1d-gaussian-mixture'], help='experiment name')
    parser.add_argument('--num_data', type=int, default=4000, help='Number of data points')
    parser.add_argument('--data_dim', type=int, default=1, help='The dimensiion of data')
    parser.add_argument('--source_name', type=str, choices=['gaussian', 'p', 'q', 'outlier', 'noise'], help='Name of source dataset')
    parser.add_argument('--target_name', type=str, required=True, choices=['gaussian', 'p', 'q', 'outlier', 'noise'], help='Name of target dataset')    
    parser.add_argument('--p', type=float, default=0., help='Only for outlier test. Fraction of outlier')
    
    # settings
    # Loss configurations
    parser.add_argument('--phi1', type=str, default='none', choices=['none', 'linear', 'kl', 'softplus', 'chi'])
    parser.add_argument('--phi2', type=str, default='none', choices=['none', 'linear', 'kl', 'softplus', 'chi'])
    parser.add_argument('--tau', type=float, help='scalar value multiplied to quadratic cost functional')
    parser.add_argument('--regularize', action='store_true', default=False, help='use regularization or not')
    parser.add_argument('--lmbda', type=float, default=0.01, help='regularization hyperparameter')

    # training configurations
    parser.add_argument('--num_epoch', type=int, default=1000, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    # kl-div
    parser.add_argument('--kl_estimator', type=str, default='scipy', help='kl divergence estimator')

    # save path
    parser.add_argument('--savepath', type=str, help='experiment save path')
    parser.add_argument('--save_every', type=int, default=100, help='Evaluation every {save_every} epoch')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--beta_min', type=float, default= 0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')
    parser.add_argument('--r1_gamma', type=float, default=2, help='coef for r1 reg')
    parser.add_argument('--version', type=str, help='experiment name')
    parser.add_argument('--time_embed_dim', type=int, default=128)

    args = parser.parse_args()
    main(args)
