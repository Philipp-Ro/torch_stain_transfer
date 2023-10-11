# -------------------------------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------------------------------
import os, sys
# add the parent and grandparent dir to be able to use the utils and eval fuction 
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
import torch.optim as optim
from U_net_model import UNet
import torchvision
import matplotlib.pyplot as plt
from BCI_UNet import UnetGenerator
import kornia
from Resnet_gen import ResnetGenerator

class model(torch.nn.Module):
    def __init__(self,params,net):
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

        self.gen = net.to(params['device'])
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))
        self.MSE_LOSS = nn.MSELoss().to(params['device'])
        self.MAE_LOSS = nn.L1Loss().to(params['device'])
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(params['device'])
        self.psnr = PeakSignalNoiseRatio().to(params['device'])
        self.params = params
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()
        self.output_folder_path = os.path.join(self.params['output_path'],self.params['output_folder'])
        self.checkpoint_folder = os.path.join(self.output_folder_path,"checkpoints")
        self.result_dir = os.path.join(self.output_folder_path,'train_result.txt')



        os.mkdir(self.checkpoint_folder)
        os.mkdir(os.path.join(os.path.join(params['output_path'],params['output_folder']),"train_plots"))

    def fit(self):
        train_eval ={}
        train_eval['mse'] = []
        train_eval['ssim'] = []
        k =0
        best_perf = 2

        for epoch in range(self.params['num_epochs']):
            mse_list = []
            ssim_list = []
                
            # the dataset is set up he coppes images out of the original image i the set size 
            # each epoch he takes a new slice of the original image 
            # recomended sizes [64,64] / [128,128] / [256, 256]  
            HE_img_dir = os.path.join(self.params['train_dir'],'HE')
            IHC_img_dir = os.path.join(self.params['train_dir'],'IHC')
           
            num_patches = (1024 * 1024) // self.params['img_size'][0]**2 
            if k>num_patches-1:
                k=0

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

                if self.params['contrast_IHC']!=0:
                    real_IHC = torchvision.transforms.functional.adjust_contrast(real_IHC,self.params['contrast_IHC'])
                # -----------------------------------------------------------------------------------------
                # Train U_net
                # -----------------------------------------------------------------------------------------
                fake_IHC = self.gen(real_HE)

                # ---------------------------- LOSS -------------------------------------------------------
                G_L2_LOSS ,fake_blurr_IHC, real_blur_IHC= self.gausian_blurr_loss(real_IHC,fake_IHC)  
                G_L3_LOSS ,fake_blurr_IHC, real_blur_IHC= self.gausian_blurr_loss(fake_blurr_IHC, real_blur_IHC)  
                G_L4_LOSS ,fake_blurr_IHC, real_blur_IHC= self.gausian_blurr_loss(fake_blurr_IHC, real_blur_IHC) 
                #fake_low,fake_high = utils.frequency_division(fake_IHC)
                #real_low,real_high = utils.frequency_division(real_IHC)
                #fft_LOSS =self.MSE_LOSS(fake_low.to(self.params['device']),real_low.to(self.params['device']))+ self.MSE_LOSS(fake_high.to(self.params['device']),real_high.to(self.params['device']))

                #fake_IHC = NormalizeTensor(fake_IHC)

                loss_gen_total = 0
                
                loss_mse = self.MSE_LOSS(real_IHC, fake_IHC)
                loss_gen = self.params['generator_lambda'] * loss_mse
                loss_gen_total = loss_gen_total + loss_gen + G_L2_LOSS + G_L3_LOSS +G_L4_LOSS

                ssim_IHC = self.ssim(fake_IHC, real_IHC)

                # ssim loss 
                if 'ssim' in self.params['total_loss_comp']:
                    loss_ssim = 1-ssim_IHC

                    loss_ssim = (self.params['ssim_lambda']*loss_ssim)
                    loss_gen_total = loss_gen_total + loss_ssim

                # psnr loss 
                if 'psnr' in self.params['total_loss_comp']:
                    psnr_IHC = self.psnr(fake_IHC, real_IHC)
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

                if (i+1) % 100 == 0:
                    train_loop.set_description(f"Epoch [{epoch+1}/{self.params['num_epochs']}]")
                    train_loop.set_postfix( Gen_loss = loss_gen_total.item())

                # saves train loss for each epoch        
                mse_list.append(loss_mse.item())
                ssim_list.append(ssim_IHC.item())

            
            # -------------------------- saving models after each 5 epochs --------------------------------
            if epoch % 5 == 0:

                utils.plot_img_set( real_HE = real_HE,
                                    fake_IHC=fake_IHC,
                                    real_IHC=real_IHC,
                                    i=i,
                                    params = self.params,
                                    img_name = img_name,
                                    step = 'train',
                                    epoch = epoch )
                # safe a checkpoint 
                epoch_name = 'gen_G_weights_'+str(epoch)
                torch.save(self.gen.state_dict(),os.path.join(self.checkpoint_folder,epoch_name ) )

            train_eval['mse'].append(np.mean(mse_list))
            train_eval['ssim'].append(np.mean(ssim_list))

            current_perf = np.mean(mse_list)+(1-np.mean(ssim_list))

            # ------- delete list to clear ram ---------------------------------------------------------
            del mse_list
            del ssim_list
            # -------- add k + 1 tchange the patches in the loader --------------------------------------
            k = k+1

            if current_perf < best_perf:
                best_perf = current_perf
                gen_out = self.gen

        # plot train results 
        x = range(self.params['num_epochs'])

        fig, axs = plt.subplots(2)
        fig.suptitle('train_results')
        axs[0].plot(x, train_eval['mse'])
        axs[0].set_title('MSE')
        axs[1].plot(x, train_eval['ssim'])
        axs[1].set_title('SSIM')

        fig.savefig(os.path.join(os.path.join(self.params['output_path'],self.params['output_folder']),"train_result.png"))



        return gen_out



    def gausian_blurr_loss(self,real_img, fake_img):
        octave1_layer2_fake=kornia.filters.gaussian_blur2d(fake_img,(3,3),(1,1))
        octave1_layer3_fake=kornia.filters.gaussian_blur2d(octave1_layer2_fake,(3,3),(1,1))
        octave1_layer4_fake=kornia.filters.gaussian_blur2d(octave1_layer3_fake,(3,3),(1,1))
        octave1_layer5_fake=kornia.filters.gaussian_blur2d(octave1_layer4_fake,(3,3),(1,1))
        octave2_layer1_fake=kornia.filters.blur_pool2d(octave1_layer5_fake, 1, stride=2)
        octave1_layer2_real=kornia.filters.gaussian_blur2d(real_img,(3,3),(1,1))
        octave1_layer3_real=kornia.filters.gaussian_blur2d(octave1_layer2_real,(3,3),(1,1))
        octave1_layer4_real=kornia.filters.gaussian_blur2d(octave1_layer3_real,(3,3),(1,1))
        octave1_layer5_real=kornia.filters.gaussian_blur2d(octave1_layer4_real,(3,3),(1,1))
        octave2_layer1_real=kornia.filters.blur_pool2d(octave1_layer5_real, 1, stride=2)
        G_L2 = self.MSE_LOSS(octave2_layer1_fake, octave2_layer1_real) 
        return G_L2,octave2_layer1_fake,octave2_layer1_real