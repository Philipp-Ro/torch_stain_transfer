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
from ViT_model import ViT_Generator
import torch.optim as optim

class model(torch.nn.Module):
    def __init__(self,params):
        super(model, self).__init__()               
        # -----------------------------------------------------------------------------------------------------------------
        # Initialize U_net
        # -----------------------------------------------------------------------------------------------------------------
        # gen transfers from domain X -> Y
        #
        # disc distinguishes between real and fake in the Y domain 
        #
        # in our case:
        # Domain X = HE
        # Domain Y = IHC
        self.gen = ViT_Generator(   chw = [params['in_channels']]+params['img_size'], 
                                    patch_size = params['patch_size'],
                                    num_heads = params['num_heads'], 
                                    num_blocks = params['num_blocks']
                                    ).to(params['device'])
        
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))
        self.MSE_LOSS = nn.MSELoss().to(params['device'])
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(params['device'])
        self.psnr = PeakSignalNoiseRatio().to(params['device'])
        self.params = params
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

    def fit(self):
        disc_loss_list = []
        gen_loss_list = []

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
                
                self.opt_gen.param_groups[0]['lr'] -= self.params['learn_rate_gen'] / (self.params['num_epochs'] - self.params['decay_epoch'])

 
            train_loop = tqdm(enumerate(train_data_loader), total = len(train_data_loader), leave= False)
            
            for i, (real_HE, real_IHC,img_name) in train_loop :
              
                # -----------------------------------------------------------------------------------------
                # Train U_net
                # -----------------------------------------------------------------------------------------
                fake_IHC = self.gen(real_HE)
                loss_gen_total = 0
                
                loss_gen = self.MSE_LOSS(real_IHC, fake_IHC.detach())

                loss_gen_total = loss_gen_total + loss_gen
                
                # denormalise images 
                unnorm_fake_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],fake_IHC)
                unnorm_real_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],real_IHC)
                
                # ssim loss 
                if 'ssim' in self.params['total_loss_comp']:
                    ssim_IHC = self.ssim(unnorm_fake_IHC, unnorm_real_IHC)
                    loss_ssim = 1-ssim_IHC

                    loss_ssim = (self.params['ssim_lambda']*loss_ssim)
                    loss_gen_total = loss_gen_total + loss_ssim

                # psnr loss 
                if 'psnr' in self.params['total_loss_comp']:
                    psnr_IHC = self.psnr(unnorm_fake_IHC, unnorm_real_IHC)
                    loss_psnr = psnr_IHC 

                    loss_psnr = (self.params['psnr_lambda']*loss_psnr)
                    loss_gen_total = loss_gen_total + loss_psnr

                if 'hist_loss' in self.params['total_loss_comp']:
                    hist_loss = utils.hist_loss(self,
                                                   real_img = real_IHC,
                                                   fake_img = fake_IHC )
                    
                    hist_loss = hist_loss*self.params['hist_lambda']
                    loss_gen_total = loss_gen_total + hist_loss

                # ------------------------- Apply Weights ---------------------------------------------------
                
                self.opt_gen.zero_grad()
                self.g_scaler.scale(loss_gen_total).backward()
                self.g_scaler.step(self.opt_gen)
                self.g_scaler.update()

                # -----------------------------------------------------------------------------------------
                # Show Progress
                # -----------------------------------------------------------------------------------------
                #saves losses in list 
                gen_loss_list.append(loss_gen_total.item())

                if (i+1) % 100 == 0:
                    train_loop.set_description(f"Epoch [{epoch+1}/{self.params['num_epochs']}]")
                    train_loop.set_postfix( Gen_loss = loss_gen_total.item())
            k = k+1
            # -------------------------- saving models after each 5 epochs --------------------------------
            if epoch % 5 == 0:
                output_folder_path = os.path.join(self.params['output_path'],self.params['output_folder'])
                epoch_name = 'gen_G_weights_'+str(epoch)

                utils.plot_img_set( real_HE = real_HE,
                                    fake_IHC=unnorm_fake_IHC,
                                    real_IHC=unnorm_real_IHC,
                                    i=i,
                                    params = self.params,
                                    img_name = img_name,
                                    step = 'train',
                                    epoch = epoch )

                torch.save(self.gen.state_dict(),os.path.join(output_folder_path,epoch_name ) )

        x=np.arange(len(gen_loss_list))
        plt.subplot(2, 1, 1)
        plt.plot(x,disc_loss_list)
        plt.title("DISC_LOSS")

        plt.subplot(2, 1, 2)
        plt.plot(x,gen_loss_list)
        plt.title("GEN_LOSS")

        plt.savefig(os.path.join(output_folder_path,'loss_graphs'))

        return self.gen 

