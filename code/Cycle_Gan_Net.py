import torch
import os
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import loader
from torch.utils.data import DataLoader
import utils
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
#from time import Timer
from tqdm import tqdm

class model(torch.nn.Module):
    def __init__(self,params, generator_G, generator_F, discriminator_X, discriminator_Y, disc_optimizer, gen_optimizer):
        super(model, self).__init__()               
        # -----------------------------------------------------------------------------------------------------------------
        # Initialize Cycle_Gan_Net
        # -----------------------------------------------------------------------------------------------------------------
        # gen_G transfers from domain X -> Y
        # gen_F transfers from domain Y -> X
        #
        # disc_X distinguishes between real and fake in the X domain 
        # disc_Y distinguishes between real and fake in the Y domain 
        #
        # in our case:
        # Domain X = HE
        # Domain Y = IHC

        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.params = params


    def fit(self):
        Tensor = torch.cuda.FloatTensor
        criterion_GAN = torch.nn.MSELoss().cuda()
        #criterion_GAN  = torch.nn.L1Loss().cuda()
        criterion_cycle = torch.nn.L1Loss().cuda()
        criterion_identity = torch.nn.L1Loss().cuda()
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        psnr = PeakSignalNoiseRatio().cuda()
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
                self.disc_optimizer.param_groups[0]['lr'] -= self.params['learn_rate_gen'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                self.gen_optimizer.param_groups[0]['lr'] -= self.params['learn_rate_gen'] / (self.params['num_epochs'] - self.params['decay_epoch'])

 
            train_loop = tqdm(enumerate(train_data_loader), total = len(train_data_loader), leave= False)
            
            for i, (real_HE, real_IHC,img_name) in train_loop :

                # Adversarial ground truths
                valid = Tensor(np.ones((real_HE.size(0)))) # requires_grad = False. Default.
                fake = Tensor(np.zeros((real_HE.size(0)))) # requires_grad = False. Default.
                
                # -----------------------------------------------------------------------------------------
                # Train Generators
                # -----------------------------------------------------------------------------------------
                
                # ------------ Generate fake_IHC and fake_HE with gen_G and gen_F -------------------------
                #
                # input shape [n, in_channels, img_size, img_size]
                # the output layer of the conv and the trans model is a nn.Tanh layer:
                # output shape [1, in_channels, img_size, img_size]
                

                fake_IHC = self.gen_G(real_HE) 
                fake_HE = self.gen_F(real_IHC)
   

                # --------------------------- Calculate losses ---------------------------------------------

                if 'gan_loss' in self.params['total_loss_comp']:
                    loss_gen_G = -1. * torch.mean(self.disc_X(fake_IHC))
                    loss_gen_F = -1. * torch.mean(self.disc_Y(fake_HE))

                    # hyper parameters
                    loss_gen_G = self.params['generator_lambda']*loss_gen_G
                    loss_gen_F = self.params['generator_lambda']*loss_gen_F


                    loss_gen_total = loss_gen_total + (loss_gen_G+ loss_gen_F)/2

                elif'wgan_loss'in self.params['total_loss_comp']:
                    wgan_loss = 0
                    loss_gen_total = loss_gen_total + wgan_loss

                elif'cgan_loss' in self.params['total_loss_comp']:
                    cgan_loss = 0
                    loss_gen_total = loss_gen_total + cgan_loss

                else :
                    print('CHOOSE gan_loss OR wgan_loss OR cgan_loss IN total_loss_comp IN THE YAML FILE' )
 
       
                # denormalise images 
                unnorm_fake_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],fake_IHC)
                unnorm_real_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],real_IHC)
                unnorm_fake_HE = utils.denomalise(self.params['mean_HE'], self.params['std_HE'],fake_HE)
                unnorm_real_HE = utils.denomalise(self.params['mean_HE'], self.params['std_HE'],real_HE)
                
                # ssim loss 
                if 'ssim' in self.params['total_loss_comp']:
                    ssim_IHC = ssim(unnorm_fake_IHC, unnorm_real_IHC)
                    loss_ssim = 1-ssim_IHC

                    loss_ssim = (self.params['ssim_lambda']*loss_ssim)
                    loss_gen_total = loss_gen_total + loss_ssim

                # psnr loss 
                if 'psnr' in self.params['total_loss_comp']:
                    psnr_IHC = psnr(unnorm_fake_IHC, unnorm_real_IHC)
                    loss_psnr = psnr_IHC 

                    loss_psnr = (self.params['psnr_lambda']*loss_psnr)
                    loss_gen_total = loss_gen_total + loss_psnr
                #


                # Identity Loss 
                loss_id_HE = criterion_identity(self.gen_F(real_IHC), real_IHC) 
                loss_id_HE = self.params['identity_lambda']*loss_id_HE
                loss_gen_F_total = loss_id_HE

                loss_id_IHC = criterion_identity(self.gen_G(real_HE), real_HE)
                loss_id_IHC = self.params['identity_lambda']*loss_id_IHC
                loss_gen_G_total = loss_gen_G_total + loss_id_IHC
                                                                             
                # Cycle Loss
                cycled_IHC = self.gen_G(fake_HE) 
                loss_cycle_G = criterion_cycle(cycled_IHC, real_IHC) 
                loss_cycle_G = self.params['cycle_lambda']*loss_cycle_G
                loss_gen_G_total = loss_gen_G_total + loss_cycle_G 
                
                cycled_HE = self.gen_F(fake_IHC)
                loss_cycle_F = criterion_cycle(cycled_HE, real_IHC)
                loss_cycle_F = self.params['cycle_lambda']*loss_cycle_F
                loss_gen_F_total = loss_gen_F_total + loss_cycle_F

                # ------------------ Combine scaled losses for gen_G and gen_F ---------------------
                loss_gen_total = loss_gen_G_total + loss_gen_F_total
                
                # ------------------------- Apply Weights ------------------------------------------

                self.gen_optimizer.zero_grad()
                loss_gen_total.backward()
                self.gen_optimizer.step()

                
                # ---------------------------------------------------------------------------------
                # Train Discriminators
                # ---------------------------------------------------------------------------------

                # --------------------------- Discriminator X -------------------------------------
                # Calculate loss

                loss_real = criterion_GAN(self.disc_X(real_HE), valid) # train to discriminate real images as real
                loss_fake = criterion_GAN(self.disc_X(fake_HE.detach()), fake) # train to discriminate fake images as fake
                
                loss_disc_X = (loss_real + loss_fake)/2

                # Apply Hyperparameter 
                loss_disc_X = self.params['disc_lambda']*loss_disc_X 
                
                # --------------------------- Discriminator Y ---------------------------------------------
                # Calculate loss
                loss_real = criterion_GAN(self.disc_Y(real_IHC), valid) # train to discriminate real images as real
                loss_fake = criterion_GAN(self.disc_Y(fake_IHC.detach()), fake) # train to discriminate fake images as fake
                
                loss_disc_Y = (loss_real + loss_fake)/2

                # Apply Hyperparameter
                loss_disc_Y = self.params['disc_lambda']*loss_disc_Y 
                

                # --------------------------- Total Discriminator Loss ------------------------------------

                loss_disc_total = (loss_disc_X + loss_disc_Y )/2

                # ------------------------- Apply Weights -------------------------------------------------
                self.disc_optimizer.zero_grad()
                loss_disc_total.backward()
                self.disc_optimizer.step()
              
                # -----------------------------------------------------------------------------------------
                # Show Progress
                # -----------------------------------------------------------------------------------------

                if (i+1) % 100 == 0:
                    train_loop.set_description(f"Epoch [{epoch+1}/{self.params['num_epochs']}]")
                    train_loop.set_postfix(loss_gen_G=loss_gen_G_total.item(), ssim = ssim_IHC.item(), MSE_gen_G = loss_gen_G.item())
            k = k+1
            # -------------------------- saving models after each 10 epochs --------------------------------
            if epoch % 10 == 0:
                output_folder_path = os.path.join(self.params['output_path'],self.params['output_folder'])
                epoch_name = 'gen_G_weights_'+str(epoch)

                torch.save(self.gen_G.state_dict(),os.path.join(output_folder_path,epoch_name ) )
              
        return self.gen_G, self.gen_F, self.disc_X, self.disc_Y



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # reset Conv2d's weight(tensor) with Gaussian Distribution
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) # reset Conv2d's bias(tensor) with Constant(0)
        elif classname.find('InstanceNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # reset BatchNorm2d's weight(tensor) with Gaussian Distribution
            torch.nn.init.constant_(m.bias.data, 0.0) # reset BatchNorm2d's bias(tensor) with Constant(0)
