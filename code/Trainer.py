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
from torchvision.models import resnet50, ResNet50_Weights 
import utils
import plot_utils
import numpy as np 
import pickle



#class train_loop(torch.nn.Module):
class train_loop:
    def __init__(self, args, model, train_plot_eval, test_plot_eval):
        super(train_loop, self).__init__()               
        # -----------------------------------------------------------------------------------------------------------------
        # Initialize Trainer
        # -----------------------------------------------------------------------------------------------------------------
        self.args = args

        # Init model and if given framework
        self.gen = model.to(args.device)
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.g_scaler = torch.cuda.amp.GradScaler()

        if self.args.gan_framework == 'pix2pix':
            self.disc = Discriminator(in_channels=args.in_channels,features=32).to(args.device)
            self.opt_disc = optim.Adam(self.disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            self.d_scaler = torch.cuda.amp.GradScaler()
        
        if self.args.gan_framework == 'pix2pix':
            self.disc  = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.disc.fc = torch.nn.Linear(self.disc.fc.in_features, 4)
            torch.nn.init.xavier_uniform_(self.disc.fc.weight)
            self.disc =  self.disc.to(args.device) 

            best_model_weights = os.path.join(Path.cwd(),'classifier_weights_IHC_256_50.pth')
            self.disc.load_state_dict(torch.load(best_model_weights))

        if args.model == 'Diffusion':
            self.diffusion = Diffusion(noise_steps=args.diff_noise_steps,img_size=args.img_size,device=args.device)  

        # Init LOSS
        self.BCE = nn.BCEWithLogitsLoss().to(args.device)
        self.MSE = nn.MSELoss().to(args.device)
        self.CE_LOSS = torch.nn.CrossEntropyLoss()
        self.L1_LOSS = nn.L1Loss().to(args.device)
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
        self.PSNR = PeakSignalNoiseRatio().to(args.device)

        self.test_plot_eval = test_plot_eval
        self.train_plot_eval = train_plot_eval
        self.count =  len(train_plot_eval ['MSE'])


    def fit(self):
        print('---------------------------------------------- ')
        print('START TRAINING')
        k =0
        best_perf = 2

        # -----------------------------------------------------------------------------------------------------------------
        # TRAIN LOOP ALL EPOCHS
        # -----------------------------------------------------------------------------------------------------------------
        for epoch in range(self.args.num_epochs):
            print('---------------------------------------------- ')
            # init lists
            mse_list = []
            ssim_list = []
            psnr_list = []
                 
            num_patches = ((1024 * 1024) // self.args.img_size**2)

            # update patch on dataloader 
            if k>num_patches-1:
                k=0

            # Train data loader
            train_data = new_loader.stain_transfer_dataset( img_patch=k, set='train',args = self.args) 
            train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False) 


            if(epoch + 1) > self.args.decay_epoch:
                self.opt_gen.param_groups[0]['lr'] -= self.args.lr / (self.args.num_epochs - self.args.decay_epoch)

            # -----------------------------------------------------------------------------------------------------------------
            # TRAIN LOOP 1 EPOCH
            # -----------------------------------------------------------------------------------------------------------------
            for i, (real_HE, real_IHC,img_name) in enumerate(train_data_loader) :

                # model prediction:
                if self.args.model == 'Diffusion':
                    # diffusion model
                    t = self.diffusion.sample_timesteps(real_IHC.shape[0]).to(self.args.device)
                    x_t, noise = self.diffusion.noise_img(real_IHC, t, real_HE)
                    noise_pred = self.gen (x_t, t)
                    total_loss = self.get_loss(real_img=noise,fake_img=noise_pred)
                    if (i+1) % 200 == 0:
                        # sample img
                        fake_IHC = self.diffusion.sample(self.gen , n=real_IHC.shape[0], y=real_HE)

                        # train eval per img:
                        mse_score = self.MSE(real_IHC,fake_IHC)
                        ssim_score = self.SSIM(real_IHC,fake_IHC)
                        psnr_score = self.PSNR(real_IHC,fake_IHC)

                        mse_list.append(mse_score.item())
                        ssim_list.append(ssim_score.item())
                        psnr_list.append(psnr_score.item())

                else:
                    # other models
                    fake_IHC = self.gen(real_HE)

                    # model loss:
                    total_loss = self.get_loss(real_img=real_IHC,fake_img=fake_IHC,img_name=img_name)

                    # train eval per img:
                    mse_score = self.MSE(real_IHC,fake_IHC)
                    ssim_score = self.SSIM(real_IHC,fake_IHC)
                    psnr_score = self.PSNR(real_IHC,fake_IHC)

                    # save group specific results
                    mse_list.append(mse_score.item())
                    ssim_list.append(ssim_score.item())
                    psnr_list.append(psnr_score.item())
                

                # model weight update
                self.opt_gen.zero_grad()
                self.g_scaler.scale(total_loss).backward()
                self.g_scaler.step(self.opt_gen)
                self.g_scaler.update()

                # discriminator update for gan models
                if self.args.gan_framework == 'pix2pix':
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
                if (i+1) % 600 == 0:
                    show_epoch  = 'Epoch: '+str(epoch)+'/'+str(self.args.num_epochs) +' | ' +str(i)+'/3396'
                    print(show_epoch)

            # save the eamn of each epoch 
            self.train_plot_eval['MSE'].append(np.mean(mse_list))
            self.train_plot_eval['SSIM'].append(np.mean(ssim_list))
            self.train_plot_eval['PSNR'].append(np.mean(psnr_list))

            self.train_plot_eval['x'].append(epoch)

            # saving checkpoints and plots every 5 epochs
            if epoch % 5 == 0:

                plot_utils.plot_img_set( real_HE = real_HE,
                                    fake_IHC = fake_IHC,
                                    real_IHC = real_IHC,
                                    save_path = self.args.tp_path,
                                    img_name = img_name,
                                    epoch = epoch + self.count )
        
                checkpoint_name = 'gen_G_weights_'+str(epoch+self.count)+'.pth'
                torch.save(self.gen.state_dict(),os.path.join(self.args.c_path,checkpoint_name ) )

                if self.args.model != "Diffusion":
                    print('- - - - - - - - - - - - - - - - - - - - - - -  ')
                    show_val  = 'Test for Epoch: '+str(epoch) 
                    print(show_val)
                    # validation every 10 epochs
                    self.test_plot_eval = self.get_test_scores( self.gen, self.test_plot_eval)
                    self.test_plot_eval['x'].append(epoch)


                    
                if self.test_plot_eval['MSE'] == []:
                    current_perf = np.mean(mse_list)+(1-np.mean(ssim_list))
                else:
                    current_perf = self.test_plot_eval['MSE'][-1]+ 1-self.test_plot_eval['SSIM'][-1]


            # add k + 1 tchange the patches in the loader 
            k = k+1

            if current_perf < best_perf:
                best_perf = current_perf
                gen_out = self.gen

        # save train_eval
        with open(os.path.join(self.args.train_path,'train_plot_eval'), "wb") as fp:   
            pickle.dump(self.train_plot_eval, fp)

        with open(os.path.join(self.args.train_path,'test_plot_eval'), "wb") as fp:   
            pickle.dump(self.test_plot_eval, fp)

        # save best model 
        final_model_path = os.path.join(self.args.train_path,'final_weights_gen.pth')
        torch.save(gen_out.state_dict(), final_model_path)

        return gen_out, self.train_plot_eval, self.test_plot_eval
    
    def get_test_scores(self, model, result):
        
        num_patches = ((1024 * 1024) // self.args.img_size**2)-1
        epoch_result =  {}
        epoch_result['MSE'] = []
        epoch_result['SSIM'] = []
        epoch_result['PSNR'] = []

        for epoch in range(0,num_patches,1):
            data_set = new_loader.stain_transfer_dataset( img_patch=epoch, set='test', args=self.args) 
            loader = DataLoader(data_set, batch_size=1, shuffle=False) 

            mse_list = []
            ssim_list = [] 
            psnr_list = []

            for i, (real_HE, real_IHC, img_name) in enumerate(loader):

                fake_IHC = model(real_HE)

                mse_score = self.MSE(real_IHC,fake_IHC)
                ssim_score = self.SSIM(real_IHC,fake_IHC)
                psnr_score = self.PSNR(real_IHC,fake_IHC)

                mse_list.append(mse_score.item())
                ssim_list.append(ssim_score.item())
                psnr_list.append(psnr_score.item())
                #mean for each epoch 
       
            epoch_result['MSE'].append(np.mean(mse_list))
            epoch_result['SSIM'].append(np.mean(ssim_list))
            epoch_result['PSNR'].append(np.mean(psnr_list))

        #mean for each epoch 
        result['MSE'].append(np.mean(epoch_result['MSE']))
        result['SSIM'].append(np.mean(epoch_result['SSIM']))
        result['PSNR'].append(np.mean(epoch_result['PSNR']))

        return result


    def get_loss(self,real_img, fake_img,img_name):
        # TO DO : SET LAMBDAS
        total_loss = 0
        if self.args.gan_framework == 'pix2pix':
            with torch.cuda.amp.autocast():
                # output for disc on fake image
                D_input_fake =torch.cat((real_img,fake_img), 1)
                D_fake = self.disc(D_input_fake.detach())
                G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
                L1_loss = self.L1_LOSS(real_img, fake_img) 
                gen_loss = G_fake_loss + L1_loss
                total_loss = total_loss+ gen_loss

        elif self.args.gan_framework == 'score_gan':
                    score = utils.get_IHC_score(img_name)

                    gt = torch.nn.functional.one_hot(score, num_classes=4)
                    gt = gt.type(torch.DoubleTensor) 
                    gt = gt.to(self.args.device)

                    outputs = self.disc(fake_img)
                    outputs = torch.squeeze(outputs)
                    GAN_loss = self.CE_LOSS(outputs, gt)  
                    L1_loss = self.L1_LOSS(real_img, fake_img) 

                    gen_loss = GAN_loss + L1_loss
                    total_loss = total_loss+ gen_loss

        else:
            gen_loss = self.MSE(real_img, fake_img)
            total_loss = total_loss+ gen_loss

        if self.args.gaus_loss:
            G_L2_LOSS ,fake_blurr_IHC, real_blur_IHC= utils.gausian_blurr_loss(self.MSE, fake_img, real_img)  
            G_L3_LOSS ,fake_blurr_IHC, real_blur_IHC= utils.gausian_blurr_loss(self.MSE, fake_blurr_IHC, real_blur_IHC)  
            G_L4_LOSS ,fake_blurr_IHC, real_blur_IHC= utils.gausian_blurr_loss(self.MSE, fake_blurr_IHC, real_blur_IHC) 
            gausian_loss = G_L2_LOSS + G_L3_LOSS +G_L4_LOSS
            gausian_loss = gausian_loss * 1
            total_loss = total_loss + gausian_loss

        if  self.args.ssim_loss:
            ssim_IHC = self.SSIM(fake_img, real_img)
            ssim_loss = 1-ssim_IHC

            ssim_loss = ssim_loss*1
            total_loss = total_loss + ssim_loss

        if  self.args.hist_loss:
            hist_loss = utils.hist_loss(self,real_img = real_img,fake_img = fake_img )
                    
            hist_loss = hist_loss*1
            loss_gen_total = loss_gen_total + hist_loss

        return total_loss

