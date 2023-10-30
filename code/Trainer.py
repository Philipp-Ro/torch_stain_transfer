# -------------------------------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------------------------------
from architectures.Discriminator_model import Discriminator
from architectures.Diffusion_model import Diffusion
import os
import torch.nn as nn
import torch
import torch.optim as optim
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from pathlib import Path
import new_loader
from torch.utils.data import DataLoader
from tqdm import tqdm
import kornia
import utils
import numpy as np 
import pickle
import matplotlib.pyplot as plt

class train_loop(torch.nn.Module):
    def __init__(self, args, model,model_name):
        super(train_loop, self).__init__()               
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
        self.args = args
        # Init model
        self.gen = model.to(args.device)
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.g_scaler = torch.cuda.amp.GradScaler()
        if self.args.gan_framework :
            self.disc = Discriminator(in_channels=args.in_channels,features=32).to(args.device)
            self.opt_disc = optim.Adam(self.disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            self.d_scaler = torch.cuda.amp.GradScaler()
        
        if self.args.diff_model:
            self.diffusion = Diffusion(noise_steps=1000,img_size=args.img_size,device=args.device)  

        # Init LOSS
        self.BCE = nn.BCEWithLogitsLoss().to(args.device)
        self.MSE_LOSS = nn.MSELoss().to(args.device)
        self.BCE = nn.BCEWithLogitsLoss().to(args.device)
        self.L1_LOSS = nn.L1Loss().to(args.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
        self.psnr = PeakSignalNoiseRatio().to(args.device)

        # inti paths 
        model_dir = os.path.join(Path.cwd(),"masterthesis_results")
        self.train_path = os.path.join(model_dir,model_name)
        self.c_path = os.path.join(self.train_path,"checkpoints")
        self.tr_path = os.path.join(self.train_path,'train_result')
        self.tp_path = os.path.join(self.train_path,'train_plots')

        self.count = 0

        if os.path.isdir(self.train_path):
            with open(self.tr_path, "rb") as fp:   # Unpickling
                self.train_eval = pickle.load(fp)
            self.count =  len(self.train_eval['train_mse'])

            # load existing model 
            best_model_weights = os.path.join(self.train_path,'final_weights_gen.pth')
            self.gen.load_state_dict(torch.load(best_model_weights))
            print('weights loaded')

        else:
            os.mkdir(self.train_path)
            os.mkdir(self.c_path)
            os.mkdir(self.tp_path)
            self.train_eval ={}
            self.train_eval['train_mse'] = []
            self.train_eval['train_ssim'] = []
            self.train_eval['train_psnr'] = []
            self.train_eval['val_mse'] = []
            self.train_eval['val_ssim'] = []
            self.train_eval['val_psnr'] = []


    def get_validation_scores(self,gen,val_data_loader):
        val_loop = tqdm(enumerate(val_data_loader), total = len(val_data_loader), leave= False)
        for i, (real_HE, real_IHC,img_name) in val_loop :
            mse_list = []
            ssim_list = []
            psnr_list = []

            fake_IHC = gen(real_HE)

            mse_score = self.MSE_LOSS(real_IHC,fake_IHC)
            ssim_score = self.ssim(real_IHC,fake_IHC)
            psnr_score = self.psnr(real_IHC,fake_IHC)

            mse_list.append(mse_score.item())
            ssim_list.append(ssim_score.item())
            psnr_list.append(psnr_score.item())

            mse_mean = np.mean(mse_list)
            ssim_mean = np.mean(ssim_list)
            psnr_mean = np.mean(psnr_list)

            return mse_mean, ssim_mean, psnr_mean


    def get_loss(self,real_img, fake_img):
        # TO DO : SET LAMBDAS
        total_loss = 0
        if self.args.gan_framework:
            with torch.cuda.amp.autocast():
                # output for disc on fake image
                D_input_fake =torch.cat((real_img,fake_img), 1)
                D_fake = self.disc(D_input_fake.detach())
                G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
                L1 = self.L1_LOSS(real_img, fake_img) *1
                gen_loss = G_fake_loss + L1
                total_loss = total_loss+ gen_loss

        else:
            gen_loss = self.MSE_LOSS(real_img, fake_img)
            total_loss = total_loss+ gen_loss

        if self.args.gaus_loss:
            G_L2_LOSS ,fake_blurr_IHC, real_blur_IHC= self.gausian_blurr_loss(real_img,fake_img)  
            G_L3_LOSS ,fake_blurr_IHC, real_blur_IHC= self.gausian_blurr_loss(fake_blurr_IHC, real_blur_IHC)  
            G_L4_LOSS ,fake_blurr_IHC, real_blur_IHC= self.gausian_blurr_loss(fake_blurr_IHC, real_blur_IHC) 
            gausian_loss = G_L2_LOSS + G_L3_LOSS +G_L4_LOSS
            gausian_loss = gausian_loss * 1
            total_loss = total_loss + gausian_loss

        if  self.args.ssim_loss:
            ssim_IHC = self.ssim(fake_img, real_img)
            ssim_loss = 1-ssim_IHC

            ssim_loss = ssim_loss*1
            total_loss = total_loss + ssim_loss

        if  self.args.hist_loss:
            hist_loss = utils.hist_loss(self,real_img = real_img,fake_img = fake_img )
                    
            hist_loss = hist_loss*1
            loss_gen_total = loss_gen_total + hist_loss

        return total_loss





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




    def plot_trainresult(self,train_eval):
        x = range(len(train_eval['train_mse']))
        fig, axs = plt.subplots(3)
        fig.suptitle('Training and Validation results')

        axs[0].plot(x, train_eval['train_mse'],label='train_mse')
        axs[0].plot(x, train_eval['val_mse'],label='val_mse')
        axs[0].legend(loc="upper right",fontsize='xx-small')
        axs[0].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
        axs[0].set_title('MSE',loc='left')

        axs[1].plot(x, train_eval['train_ssim'],label='train_ssim')
        axs[1].plot(x, train_eval['val_ssim'],label='val_ssim')
        axs[1].legend(loc="lower right",fontsize='xx-small')
        axs[1].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
        axs[1].set_title('SSIM',loc='left')

        axs[2].plot(x, train_eval['train_psnr'],label='train_psnr')
        axs[2].plot(x, train_eval['val_psnr'],label='val_psnr')
        axs[2].legend(loc="lower right",fontsize='xx-small')
        axs[2].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
        axs[2].set_title('PSNR',loc='left')
        
        plt.subplots_adjust(hspace=1.3)
        fig.savefig(os.path.join(self.train_path,"train_result.png"))



    def fit(self):
        

        k =0
        best_perf = 2

        for epoch in range(self.args.num_epochs):
            mse_list = []
            ssim_list = []
            psnr_list = []
                
            # the dataset is set up he coppes images out of the original image i the set size 
            # each epoch he takes a new slice of the original image 
            # recomended sizes [64,64] / [128,128] / [256, 256]  
           
            num_patches = (1024 * 1024) // self.args.img_size**2 
            if k>num_patches-2:
                k=0

            # Train data loader
            train_data = new_loader.stain_transfer_dataset( img_patch=k, set='train',args = self.args) 
            train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False) 

            # Val data loader
            val_data = new_loader.stain_transfer_dataset( img_patch=num_patches-1, set='train',args = self.args)  
            val_data_loader = DataLoader(val_data, batch_size=1, shuffle=False)

            if(epoch + 1) > self.args.decay_epoch:
                self.opt_gen.param_groups[0]['lr'] -= self.args.lr / (self.args.num_epochs - self.args.decay_epoch)

 
            train_loop = tqdm(enumerate(train_data_loader), total = len(train_data_loader), leave= False)
            
            for i, (real_HE, real_IHC,img_name) in train_loop :
                # -----------------------------------------------------------------------------------------
                # Train Model
                # -----------------------------------------------------------------------------------------
                if self.args.diff_model:
                    # diffusion model
                    t = self.diffusion.sample_timesteps(real_IHC.shape[0]).to(self.args.device)
                    x_t, noise = self.diffusion.noise_img(real_IHC, t, real_HE)
                    noise_pred = self.gen (x_t, t)
                    total_loss = self.get_loss(real_img=noise,fake_img=noise_pred)
                    if (i+1) % 150 == 0:
                        # sample img
                        fake_IHC = self.diffusion.sample(self.gen , n=real_IHC.shape[0], y=real_HE)

                        # train eval per img:
                        mse_score = self.MSE_LOSS(real_IHC,fake_IHC)
                        ssim_score = self.ssim(real_IHC,fake_IHC)
                        psnr_score = self.psnr(real_IHC,fake_IHC)

                        mse_list.append(mse_score.item())
                        ssim_list.append(ssim_score.item())
                        psnr_list.append(psnr_score.item())


                else:
                    # predict with generator:
                    fake_IHC = self.gen(real_HE)

                    # generator loss:
                    total_loss = self.get_loss(real_img=real_IHC,fake_img=fake_IHC)

                    # train eval per img:
                    mse_score = self.MSE_LOSS(real_IHC,fake_IHC)
                    ssim_score = self.ssim(real_IHC,fake_IHC)
                    psnr_score = self.psnr(real_IHC,fake_IHC)

                    mse_list.append(mse_score.item())
                    ssim_list.append(ssim_score.item())
                    psnr_list.append(psnr_score.item())

                # generator weight update
                self.opt_gen.zero_grad()
                self.g_scaler.scale(total_loss).backward()
                self.g_scaler.step(self.opt_gen)
                self.g_scaler.update()

                # discriminator update for gan models
                if self.args.gan_framework:
                    with torch.cuda.amp.autocast():
                        D_input_fake =torch.cat((real_HE,fake_IHC), 1)
                        D_input_real =torch.cat((real_HE,real_IHC), 1)

                        D_real = self.disc(D_input_real.detach())
                        D_real_loss = self.BCE(D_real, torch.ones_like(D_real))

                        D_fake = self.disc(D_input_fake.detach())
                        D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))

                        
                        loss_disc = (D_real_loss + D_fake_loss) / 2

                        self.disc.zero_grad()
                        self.d_scaler.scale(loss_disc).backward()
                        self.d_scaler.step(self.opt_disc)
                        self.d_scaler.update()
                

                # show progress:
                if (i+1) % 100 == 0:
                    train_loop.set_description(f"Epoch [{epoch+1}/{self.args.num_epochs}]")
                    train_loop.set_postfix( Gen_loss = total_loss.item())



            # train eval per epoch:
            self.train_eval['train_mse'].append(np.mean(mse_list))
            self.train_eval['train_ssim'].append(np.mean(ssim_list))
            self.train_eval['train_psnr'].append(np.mean(psnr_list))

            # validation
            val_mse, val_ssim, val_psnr = self.get_validation_scores(self.gen,val_data_loader)
            self.train_eval['val_mse'].append(val_mse)
            self.train_eval['val_ssim'].append(val_ssim)
            self.train_eval['val_psnr'].append(val_psnr)
                
            # saving checkpoints and plots every 5 epochs
            if epoch % 5 == 0:

                utils.plot_img_set( real_HE = real_HE,
                                    fake_IHC = fake_IHC,
                                    real_IHC = real_IHC,
                                    save_path = self.tp_path,
                                    img_name = img_name,
                                    epoch = epoch + self.count )
        
                checkpoint_name = 'gen_G_weights_'+str(epoch+self.count)+'.pth'
                torch.save(self.gen.state_dict(),os.path.join(self.c_path,checkpoint_name ) )





            # add k + 1 tchange the patches in the loader 
            k = k+1

            # update best model
            current_perf =val_mse+(1-val_ssim)

            if current_perf < best_perf:
                best_perf = current_perf
                gen_out = self.gen

        # save train_eval
        with open(self.tr_path, "wb") as fp:   
            pickle.dump(self.train_eval, fp)

        # train metric plots
        self.plot_trainresult(self.args,self.train_eval)

        # save best model 
        final_model_path = os.path.join(self.train_path,'final_weights_gen.pth')
        torch.save(gen_out.state_dict(), final_model_path)

        return gen_out, self.train_eval

