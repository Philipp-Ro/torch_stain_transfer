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
        self.criterion_GAN = torch.nn.BCELoss().cuda()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.psnr = PeakSignalNoiseRatio().cuda()


    def fit(self):
        Tensor = torch.cuda.FloatTensor

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
                self.valid = Tensor(np.ones((real_HE.size(0)))) # requires_grad = False. Default.
                self.fake = Tensor(np.zeros((real_HE.size(0)))) # requires_grad = False. Default.
                
                # -----------------------------------------------------------------------------------------
                # Train Generator
                # -----------------------------------------------------------------------------------------
                fake_IHC = self.gen(real_HE) 
                loss_gen_total = 0

                loss_gen = utils.generator_loss(self, 
                                                disc = self.disc,
                                                fake_img = fake_IHC,
                                                params = self.params)
                
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

                # ------------------------- Apply Weights ---------------------------------------------------
                
                self.gen_optimizer.zero_grad()
                loss_gen_total.backward()
                self.gen_optimizer.step()


  
                # ---------------------------------------------------------------------------------
                # Train Discriminator
                # ---------------------------------------------------------------------------------
                
                if 'gan_loss' in self.params['total_loss_comp']:
                    # vanilla gan with the loss function of goodfellow
                    loss_disc = utils.discriminator_loss(   self, 
                                                            disc = self.disc, 
                                                            real_img = real_IHC, 
                                                            fake_img = fake_IHC, 
                                                            params = self.params)
                    
                    

                # ------------------------- Apply Weights ---------------------------------------------------
                    loss_disc_print = loss_disc

                    self.disc_optimizer.zero_grad()
                    loss_disc.backward()
                    self.disc_optimizer.step()
                
                elif'wgan_loss'in self.params['total_loss_comp']:
                    for d_iter in range(self.params['disc_iterations']):
                        gp = utils.gradient_penalty(self.disc, real_IHC, fake_IHC)

                        loss_critic = utils.discriminator_loss( self, 
                                                                disc = self.disc, 
                                                                real_img = real_IHC, 
                                                                fake_img = fake_IHC, 
                                                                params = self.params)
                        
                        loss_critic = loss_critic + self.params['disc_lambda']*gp.detach() 
                        
                # ------------------------- Apply Weights ---------------------------------------------------
                        loss_disc_print = loss_critic

                        self.disc.zero_grad()
                        loss_critic.backward()
                        self.disc_optimizer.step()
            
                # -----------------------------------------------------------------------------------------
                # Show Progress
                # -----------------------------------------------------------------------------------------

                if (i+1) % 100 == 0:
                    train_loop.set_description(f"Epoch [{epoch+1}/{self.params['num_epochs']}]")
                    train_loop.set_postfix( Gen_loss = loss_gen_total, disc_loss = loss_disc_print)
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
                

