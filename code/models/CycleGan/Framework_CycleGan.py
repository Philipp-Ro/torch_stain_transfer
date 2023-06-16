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
from U_net_Generator_model import U_net_Generator
from Vanilla_Discriminator_model import Discriminator
import torch.optim as optim

class model(torch.nn.Module):
    def __init__(self,params):
        super(model, self).__init__()               
        # -----------------------------------------------------------------------------------------------------------------
        # Initialize CycleGan
        # -----------------------------------------------------------------------------------------------------------------
        # gen transfers from domain X -> Y
        #
        # disc distinguishes between real and fake in the Y domain 
        #
        # in our case:
        # Domain X = HE
        # Domain Y = IHC
        self.disc_X = Discriminator(in_channels=params['in_channels'],features=params['disc_features']).to(params['device'])
        self.disc_Y = Discriminator(in_channels=params['in_channels'],features=params['disc_features']).to(params['device'])
        self.gen_G = U_net_Generator(in_channels=params['in_channels'], features=params['gen_features']).to(params['device'])
        self.gen_F = U_net_Generator(in_channels=params['in_channels'], features=params['gen_features']).to(params['device'])

        self.opt_disc = optim.Adam(
                                    list(self.disc_X.parameters()) + list(self.disc_Y.parameters()),
                                    lr=params['learn_rate_disc'],
                                    betas=(params['beta1'],params['beta2']),
                                )

        self.opt_gen = optim.Adam(
                                    list(self.gen_G.parameters()) + list(self.gen_F.parameters()),
                                    lr=params['learn_rate_gen'],
                                    betas=(params['beta1'],params['beta2']),
                                )

        self.MSE_LOSS = nn.MSELoss().to(params['device'])
        self.L1_LOSS = nn.L1Loss().to(params['device'])
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
                self.opt_disc.param_groups[0]['lr'] -= self.params['learn_rate_gen'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                self.opt_gen.param_groups[0]['lr'] -= self.params['learn_rate_gen'] / (self.params['num_epochs'] - self.params['decay_epoch'])

 
            train_loop = tqdm(enumerate(train_data_loader), total = len(train_data_loader), leave= False)
            
            for i, (real_HE, real_IHC,img_name) in train_loop :
              
                # -----------------------------------------------------------------------------------------
                # Train Generator
                # -----------------------------------------------------------------------------------------
                # ------------ Generate fake_IHC and fake_HE with gen_G and gen_F -------------------------
                #
                # input shape [n, in_channels, img_size, img_size]
                # the output layer of the conv and the trans model is a nn.Tanh layer:
                # output shape [1, in_channels, img_size, img_size]
                                
                loss_gen_total = 0

                fake_IHC = self.gen_G(real_HE).detach()
                fake_HE = self.gen_F(real_IHC).detach()

                with torch.cuda.amp.autocast():
                    # -------------------- adversarial loss for both generators ----------------------------
                    D_IHC_fake = self.disc_Y(fake_IHC)
                    D_HE_fake = self.disc_X(fake_HE)
                    loss_gen_G = self.MSE_LOSS(D_IHC_fake, torch.ones_like(D_IHC_fake))
                    loss_gen_F = self.MSE_LOSS(D_HE_fake, torch.ones_like(D_HE_fake))
                    # hyper Parameter
                    loss_gen = loss_gen_G + loss_gen_F
                    loss_gen = loss_gen * self.params['generator_lambda']


                    # -------------------- cycle loss ------------------------------------------------------
                    cycle_IHC = self.gen_G(fake_HE)
                    cycle_HE = self.gen_F(fake_IHC)
                    cycle_IHC_loss = self.L1_LOSS(real_IHC, cycle_IHC)
                    cycle_HE_loss = self.L1_LOSS(real_HE, cycle_HE)
                    # hyper Parameter
                    loss_cycle = cycle_IHC_loss + cycle_HE_loss
                    loss_cycle = loss_cycle * self.params['cycle_lambda']


                    # -------------------- identity loss ---------------------------------------------------
                    identity_IHC = self.gen_G(real_IHC)
                    identity_HE = self.gen_F(real_HE)
                    identity_IHC_loss = self.L1_LOSS(real_IHC, identity_IHC)
                    identity_HE_loss = self.L1_LOSS(real_HE, identity_HE)
                    # hyper Parameter
                    loss_identity = identity_IHC_loss + identity_HE_loss
                    loss_identity = loss_identity * self.params['identity_lambda']

                    loss_gen_total = loss_gen + loss_cycle + loss_identity 

                #  denormalise images 
                unnorm_fake_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],fake_IHC)
                unnorm_real_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],real_IHC)
                unnorm_fake_HE = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],fake_HE)
                unnorm_real_HE = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],real_HE)
        
                # ------------------- ssim loss -------------------------------------------------------- 
                if 'ssim' in self.params['total_loss_comp']:
                    ssim_IHC = self.ssim(unnorm_fake_IHC, unnorm_real_IHC)
                    loss_ssim_IHC = 1-ssim_IHC

                    ssim_HE = self.ssim(unnorm_fake_HE, unnorm_real_HE)
                    loss_ssim_HE = 1-ssim_HE

                    # hyper Parameter
                    loss_ssim = loss_ssim_IHC + loss_ssim_HE
                    loss_ssim = (self.params['ssim_lambda']*loss_ssim)
                    loss_gen_total = loss_gen_total + loss_ssim

                # ----------------- psnr loss ---------------------------------------------------------- 
                if 'psnr' in self.params['total_loss_comp']:
                    psnr_IHC = self.psnr(unnorm_fake_IHC, unnorm_real_IHC)
                    loss_psnr_IHC = psnr_IHC 

                    psnr_IHC = self.psnr(unnorm_fake_HE, unnorm_real_HE)
                    loss_psnr_HE = psnr_IHC 

                    # hyper Parameter
                    loss_ssim = loss_psnr_IHC + loss_psnr_HE
                    loss_psnr = (self.params['psnr_lambda']*loss_psnr)
                    loss_gen_total = loss_gen_total + loss_psnr

                # --------------- hist_loss ------------------------------------------------------------
                if 'hist_loss' in self.params['total_loss_comp']:
                    hist_loss_HE = utils.hist_loss(self,
                                                    real_img = real_HE,
                                                    fake_img= fake_HE )
                        
                    hist_loss_IHC = utils.hist_loss(self,
                                                    real_img = real_IHC,
                                                    fake_img= fake_IHC )
                        
                    hist_loss = hist_loss_HE + hist_loss_IHC
                    hist_loss = hist_loss*self.params['hist_lambda']
                        
                    loss_gen_total = loss_gen_total + hist_loss

                # ------------------------- Apply Weights ----------------------------------------------
                loss_gen_print = loss_gen_total
                self.opt_gen.zero_grad()
                self.g_scaler.scale(loss_gen_total).backward()
                self.g_scaler.step(self.opt_gen)
                self.g_scaler.update()

                # ---------------------------------------------------------------------------------
                # Train Discriminator
                # ---------------------------------------------------------------------------------
                with torch.cuda.amp.autocast():
                    
                    D_real_IHC = self.disc_Y(real_IHC)
                    D_real_loss_IHC = self.MSE_LOSS(D_real_IHC, torch.ones_like(D_real_IHC))
                    D_fake_IHC = self.disc_Y(fake_IHC.detach())
                    D_fake_loss_IHC = self.MSE_LOSS(D_fake_IHC, torch.zeros_like(D_fake_IHC))
                    loss_disc_IHC = D_real_loss_IHC + D_fake_loss_IHC

                    D_real_HE = self.disc_X(real_HE)
                    D_real_loss_HE = self.MSE_LOSS(D_real_HE, torch.ones_like(D_real_HE))
                    D_fake_HE = self.disc_X(fake_HE.detach())
                    D_fake_loss_HE = self.MSE_LOSS(D_fake_HE, torch.zeros_like(D_fake_HE))
                    loss_disc_HE = D_real_loss_HE + D_fake_loss_HE
                    
                    loss_disc = (loss_disc_IHC + loss_disc_HE)/2
                    loss_disc_print = loss_disc

                    self.opt_disc.zero_grad()
                    self.d_scaler.scale(loss_disc).backward()
                    self.d_scaler.step(self.opt_disc)
                    self.d_scaler.update()

                # -----------------------------------------------------------------------------------------
                # Show Progress
                # -----------------------------------------------------------------------------------------
                #saves losses in list 
                disc_loss_list.append(loss_disc_print.item())
                gen_loss_list.append(loss_gen_total.item())

                if (i+1) % 100 == 0:
                    train_loop.set_description(f"Epoch [{epoch+1}/{self.params['num_epochs']}]")
                    train_loop.set_postfix( Gen_loss = loss_gen_total.item(), disc_loss = loss_disc_print.item())
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

                torch.save(self.gen_G.state_dict(),os.path.join(output_folder_path,epoch_name ) )

        x=np.arange(len(gen_loss_list))
        plt.subplot(2, 1, 1)
        plt.plot(x,disc_loss_list)
        plt.title("DISC_LOSS")

        plt.subplot(2, 1, 2)
        plt.plot(x,gen_loss_list)
        plt.title("GEN_LOSS")

        plt.savefig(os.path.join(output_folder_path,'loss_graphs'))

        return self.gen_G, self.disc_Y
