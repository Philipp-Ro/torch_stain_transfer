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
import torch.nn as nn
import torch.optim as optim
import kornia
import pickle

class model(torch.nn.Module):
    def __init__(self,params, net, epoch_path, train_eval):
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
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(params['device'])
        self.psnr = PeakSignalNoiseRatio().to(params['device'])
        self.params = params
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()
        self.epoch_path = epoch_path
        self.train_eval = train_eval

    def fit(self):
        train_result_dir = os.path.join(self.epoch_path,'train_result')
        checkpoint_folder = os.path.join(self.epoch_path,"checkpoints")
        trainplots_folder = os.path.join(self.epoch_path,"train_plots")
        os.mkdir(checkpoint_folder)
        os.mkdir(trainplots_folder)

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
                k=1

            train_data = loader.stain_transfer_dataset( img_patch=  k,
                                                        img_size= self.params['img_size'],
                                                        HE_img_dir = HE_img_dir,
                                                        IHC_img_dir = IHC_img_dir,    
                                                        params=self.params                                                 
                                           )
            
            # get dataloader
            train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False) 

            if(epoch + 1) > self.params['decay_epoch']:
                
                self.opt_gen.param_groups[0]['lr'] -= self.params['learn_rate_gen'] / (self.params['num_epochs'] - self.params['decay_epoch'])

 
            train_loop = tqdm(enumerate(train_data_loader), total = len(train_data_loader), leave= False)
            
            for i, (real_HE, real_IHC,img_name) in train_loop :
              
                # -----------------------------------------------------------------------------------------
                # Train ViT
                # -----------------------------------------------------------------------------------------
                fake_IHC = self.gen(real_HE)
                loss_gen_total = 0
                
                loss_mse = self.MSE_LOSS(real_IHC, fake_IHC)
                loss_gen = self.params['generator_lambda'] * loss_mse
                loss_gen_total = loss_gen_total + loss_gen

                ssim_IHC = self.ssim(fake_IHC, real_IHC)

                if "normalise" in self.params["preprocess_IHC"]:
                    # denormalise images 
                    fake_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],fake_IHC)
                    real_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],real_IHC)

                if 'mse_color' in self.params['total_loss_comp']:
                    mse_color_list = []
                    fake_IHC_0 = fake_IHC[:,0,:,:]
                    fake_IHC_1 = fake_IHC[:,1,:,:]
                    fake_IHC_2 = fake_IHC[:,2,:,:]

                    real_IHC_0 = real_IHC[:,0,:,:]
                    real_IHC_1 = real_IHC[:,1,:,:]
                    real_IHC_2 = real_IHC[:,2,:,:]

                    mse_color_list.append(self.MSE_LOSS(fake_IHC_0 ,real_IHC_0))
                    mse_color_list.append(self.MSE_LOSS(fake_IHC_1 ,real_IHC_1))
                    mse_color_list.append(self.MSE_LOSS(fake_IHC_2 ,real_IHC_2))

                    mse_color = max(mse_color_list)
                    loss_gen_total = loss_gen_total + mse_color

                if 'gausian_loss' in self.params['total_loss_comp']:
                    G_L2_LOSS ,fake_blurr_IHC, real_blur_IHC= self.gausian_blurr_loss(real_IHC,fake_IHC)  
                    G_L3_LOSS ,fake_blurr_IHC, real_blur_IHC= self.gausian_blurr_loss(fake_blurr_IHC, real_blur_IHC)  
                    G_L4_LOSS ,fake_blurr_IHC, real_blur_IHC= self.gausian_blurr_loss(fake_blurr_IHC, real_blur_IHC) 
                    loss_gausian = G_L2_LOSS + G_L3_LOSS +G_L4_LOSS
                    loss_gen_total = loss_gen_total + loss_gausian
                
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

                mse_list.append(loss_mse.item())
                ssim_list.append(ssim_IHC.item())

            # -------------------------- saving models after each 5 epochs --------------------------------
            if epoch % 5 == 0:
                utils.plot_img_set( real_HE = real_HE,
                                    fake_IHC=fake_IHC,
                                    real_IHC=real_IHC,
                                    save_path = trainplots_folder,
                                    img_name = img_name,
                                    epoch = epoch )
                # safe a checkpoint 
                epoch_name = 'gen_G_weights_'+str(epoch)
                torch.save(self.gen.state_dict(),os.path.join(checkpoint_folder,epoch_name ) )

            self.train_eval['mse'].append(np.mean(mse_list))
            self.train_eval['ssim'].append(np.mean(ssim_list))

            current_perf = np.mean(mse_list)+(1-np.mean(ssim_list))

            # ------- delete list to clear ram ---------------------------------------------------------
            del mse_list
            del ssim_list
            # -------- add k + 1 tchange the patches in the loader --------------------------------------
            k = k+1

            if current_perf < best_perf:
                best_perf = current_perf
                gen_out = self.gen

        with open(train_result_dir, "wb") as fp:   #Pickling
            pickle.dump(self.train_eval, fp)

       
        return gen_out,self.train_eval


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
