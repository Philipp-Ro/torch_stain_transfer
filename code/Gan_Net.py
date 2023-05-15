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
    def __init__(self,params, generator, discriminator, disc_optimizer, gen_optimizer):
        super(model, self).__init__()               
        # -----------------------------------------------------------------------------------------------------------------
        # Initialize Gan_Net
        # -----------------------------------------------------------------------------------------------------------------
        # gen transfers from domain X -> Y
        #
        # disc distinguishes between real and fake in the Y domain 
        #
        # in our case:
        # Domain X = HE
        # Domain Y = IHC

        self.gen = generator
        self.disc = discriminator
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.params = params
        self.sigmoid = torch.nn.Sigmoid()


    def fit(self):
        Tensor = torch.cuda.FloatTensor
        #-------------------- loss functions and metrics ------------------------------
        #  check out https://neptune.ai/blog/gan-loss-functions
        #criterion_GAN = torch.nn.MSELoss().cuda()
        criterion_GAN = torch.nn.BCELoss().cuda()
        # criterion_GAN  = torch.nn.L1Loss().cuda()
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
                # Train Generator
                # -----------------------------------------------------------------------------------------
                
                # ------------ Generate fake_IHC with gen -------------------------------------------------
                #
                # input shape [n, in_channels, img_size, img_size]
                # the output layer of the conv and the trans model is a nn.Tanh layer:
                # output shape [1, in_channels, img_size, img_size]
                

                fake_IHC = self.gen(real_HE) 
                loss_gen_total = 0
                # --------------------------- Calculate losses ---------------------------------------------
                # Generator loss
                if 'gan_loss' in self.params['total_loss_comp']:
                    disc_pred_fake = self.disc(fake_IHC).flatten()
                    disc_probablity_fake = self.sigmoid(disc_pred_fake)

                    loss_gan = criterion_GAN(disc_probablity_fake, valid) 
                    loss_gan = self.params['generator_lambda']*loss_gan

                    loss_gen_total = loss_gen_total + loss_gan

                elif'wgan_loss'in self.params['total_loss_comp']:
                    wgan_loss = -1. * torch.mean(self.disc(fake_IHC.detach()))
                    loss_gen_total = loss_gen_total + wgan_loss


                else :
                    print('CHOOSE gan_loss OR wgan_loss  IN total_loss_comp IN THE YAML FILE' )
 
               
               
                # denormalise images 
                unnorm_fake_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],fake_IHC)
                unnorm_real_IHC = utils.denomalise(self.params['mean_IHC'], self.params['std_IHC'],real_IHC)
                
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

                # ------------------------- Apply Weights ---------------------------------------------------
                self.gen_optimizer.zero_grad()
                loss_gen_total.backward()
                self.gen_optimizer.step()

                # ---------------------------------------------------------------------------------
                # Train Discriminators
                # ---------------------------------------------------------------------------------
                
                # Calculate loss
                if 'gan_loss' in self.params['total_loss_comp']:
  
                    disc_pred_real = self.disc(real_IHC).flatten()
                    disc_probablity_real = self.sigmoid(disc_pred_real)

                    loss_real = criterion_GAN(disc_probablity_real, valid) # train to discriminate real images as real 
                    loss_real = loss_real *self.params['disc_lambda']
                    loss_real.backward()

                    loss_fake = criterion_GAN(disc_probablity_fake, fake) # train to discriminate fake images as fake
                    loss_fake = loss_fake**self.params['disc_lambda']
                    loss_fake.backward
                    
                elif'wgan_loss'in self.params['total_loss_comp']:

                    for d_iter in range(self.params['disc_iterations']):

                        loss_disc  = -(torch.mean(self.disc(fake_IHC.flatten())) - torch.mean(self.disc(real_IHC)))

                    
                    loss_disc = self.params['disc_lambda']*loss_disc 
                
                # ------------------------- Discriminator step --------------------------------------------

                self.disc_optimizer.zero_grad()
                self.disc_optimizer.step()

                # -----------------------------------------------------------------------------------------
                # Show Progress
                # -----------------------------------------------------------------------------------------

                if (i+1) % 100 == 0:
                    train_loop.set_description(f"Epoch [{epoch+1}/{self.params['num_epochs']}]")
                    train_loop.set_postfix(loss_gen=loss_gen_total.item(), ssim = ssim_IHC.item(), MSE_gen = loss_gan.item())
            k = k+1
            # -------------------------- saving models after each 10 epochs --------------------------------
            if epoch % 10 == 0:
                output_folder_path = os.path.join(self.params['output_path'],self.params['output_folder'])
                epoch_name = 'gen_G_weights_'+str(epoch)

                torch.save(self.gen.state_dict(),os.path.join(output_folder_path,epoch_name ) )
              
        return self.gen, self.disc



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # reset Conv2d's weight(tensor) with Gaussian Distribution
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) # reset Conv2d's bias(tensor) with Constant(0)
        elif classname.find('InstanceNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # reset BatchNorm2d's weight(tensor) with Gaussian Distribution
            torch.nn.init.constant_(m.bias.data, 0.0) # reset BatchNorm2d's bias(tensor) with Constant(0)
                
