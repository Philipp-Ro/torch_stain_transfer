# -------------------------------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------------------------------
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)

import loader 
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import utils
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from Diffusion_model import Diffusion
import torch.optim as optim
from U_net_simpel_diff import U_net_Generator
from Unet_diff import UNet

class model(torch.nn.Module):
    def __init__(self,params):
        super(model, self).__init__()               
        # -----------------------------------------------------------------------------------------------------------------
        # Diffusion model
        # -----------------------------------------------------------------------------------------------------------------
        # gen transfers from domain X -> Y
        #
        # disc distinguishes between real and fake in the Y domain 
        #
        # in our case:
        # Domain X = HE
        # Domain Y = IHC
        self.diffusion = Diffusion(noise_steps=params['noise_steps'],beta_start=params['beta_start'],beta_end=params['beta_end'],img_size=params['img_size'],device=params['device'])  
        self.U_net = UNet().to(params['device']) 
        #self.U_net = U_net_Generator(in_channels=params['in_channels'], features=params['U_net_features']).to(params['device'])
        self.opt_U_net = optim.Adam(self.U_net.parameters(), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))
        self.MSE_LOSS = nn.MSELoss().to(params['device'])
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(params['device'])
        self.psnr = PeakSignalNoiseRatio().to(params['device'])
        self.params = params
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

    def fit(self):
        diffusion_loss_list = []

        k =0
        for epoch in range(self.params['num_epochs']):
                
            # the dataset is set up he coppes images out of the original image i the set size 
            # each epoch he takes a new slice of the original image 
            # recomended sizes [64,64] / [128,128] / [256, 256]  
            HE_img_dir = os.path.join(self.params['train_dir'],'HE')
            IHC_img_dir = os.path.join(self.params['train_dir'],'IHC')
           
            num_patches = (1024 * 1024) // self.params['img_size'][0]**2 
            if k>num_patches-1:
                k=1

            train_data = loader.stain_transfer_dataset( img_patch=  k,
                                                        params= self.params,
                                                        HE_img_dir = HE_img_dir,
                                                        IHC_img_dir = IHC_img_dir,                                                     
                                           )
            
            # get dataloader
            train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False) 

            if(epoch + 1) > self.params['decay_epoch']:
                
                self.opt_U_net.param_groups[0]['lr'] -= self.params['learn_rate_gen'] / (self.params['num_epochs'] - self.params['decay_epoch'])

 
            train_loop = tqdm(enumerate(train_data_loader), total = len(train_data_loader), leave= False)
            
            for i, (real_HE, real_IHC,img_name) in train_loop :
              
                # -----------------------------------------------------------------------------------------
                # Train Diffusion model
                # -----------------------------------------------------------------------------------------
                self.opt_U_net.zero_grad()
                #print(real_IHC.shape)
                #t = torch.randint(0, self.params['noise_steps'], (real_HE.shape[0],), device=self.params['device']).long()
                t = self.diffusion.sample_timesteps(real_IHC.shape[0]).to(self.params['device'])
                x_t, noise = self.diffusion.noise_img(real_IHC, t)

                
                #x_t, noise = self.diffusion.forward_diffusion_sample(x_0=real_IHC, t=t)
                #noise_pred = self.U_net(x_t, t)
                noise_pred = self.U_net (x_t, t)
                diffusion_loss = self.MSE_LOSS(noise, noise_pred)

                
                diffusion_loss.backward()
                self.opt_U_net.step()

                # -----------------------------------------------------------------------------------------
                # Show Progress
                # -----------------------------------------------------------------------------------------
                #saves losses in list 
                diffusion_loss_list.append(diffusion_loss.item())

                if (i+1) % 100 == 0:
                    train_loop.set_description(f"Epoch [{epoch+1}/{self.params['num_epochs']}]")
                    train_loop.set_postfix( Gen_loss = diffusion_loss.item())
            k = k+1
            # -------------------------- saving models after each 5 epochs --------------------------------
            if epoch % 5 == 0:
                output_folder_path = os.path.join(self.params['output_path'],self.params['output_folder'])
                epoch_name = 'gen_G_weights_'+str(epoch)
                sampled_images = self.diffusion.sample(self.U_net , n=real_IHC.shape[0])

                utils.plot_img_set( real_HE = real_HE,
                                    fake_IHC=sampled_images,
                                    real_IHC=real_IHC,
                                    i=i,
                                    params = self.params,
                                    img_name = img_name,
                                    step = 'train',
                                    epoch = epoch )

                torch.save(self.U_net.state_dict(),os.path.join(output_folder_path,epoch_name ) )

        x=np.arange(len(diffusion_loss_list))

        plt.plot(x,diffusion_loss_list)
        plt.title("U_net_LOSS")

        plt.savefig(os.path.join(output_folder_path,'loss_graphs'))

        return self.U_net