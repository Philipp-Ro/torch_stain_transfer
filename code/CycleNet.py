import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import loader
from torch.utils.data import DataLoader
import utils
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio

class model(torch.nn.Module):
    def __init__(self,params, generator_G, generator_F, discriminator_X, discriminator_Y, disc_X_optimizer, disc_Y_optimizer, gen_optimizer):
        super(model, self).__init__()
        # initialse the generator G and F
        # gen_F transfers from doman Y -> X
        # gen_G transfers from doman X -> Y
        # 
        # initialize the discriminator X and Y
        # disc_X distinguishes between real and fake in the X domain 
        # disc_Y distinguishes between real and fake in the Y domain 
        #
        #
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.gen_optimizer = gen_optimizer
        self.params = params


    def fit(self):
        Tensor = torch.cuda.FloatTensor
        criterion_GAN = torch.nn.MSELoss().cuda()
        criterion_cycle = torch.nn.L1Loss().cuda()
        criterion_identity = torch.nn.L1Loss().cuda()
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        psnr = PeakSignalNoiseRatio().cuda()
        
        for epoch in range(self.params['num_epochs']):

            # the dataset is set up he coppes images out of the original image i the set size 
            # each epoch he takes a new slice of the original image 
            # recomended sizes [64,64] / [128,128] / [256, 256]  
            train_path = self.params['train_dir']
            HE_img_dir = "{}{}".format(train_path,'/HE_imgs/HE')
            IHC_img_dir = "{}{}".format(train_path,'/IHC_imgs/IHC')

            train_data = loader.stain_transfer_dataset( epoch = epoch,
                                                        num_epochs = self.params['num_epochs'],
                                                        HE_img_dir = HE_img_dir,
                                                        IHC_img_dir = IHC_img_dir,
                                                        img_size= self.params['img_size'],
                                           )

            # get dataloader
            train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False) 

            if(epoch + 1) > self.params['decay_epoch']:
                self.disc_X_optimizer.param_groups[0]['lr'] -= self.params['learn_rate_disc'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                self.disc_Y_optimizer.param_groups[0]['lr'] -= self.params['learn_rate_disc'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                self.gen_optimizer.param_groups[0]['lr'] -= self.params['learn_rate_gen'] / (self.params['num_epochs'] - self.params['decay_epoch'])
            
            for i, (real_HE, real_IHC) in enumerate(train_data_loader):

                # Adversarial ground truths
                valid = Tensor(np.ones((real_HE.size(0)))) # requires_grad = False. Default.
                fake = Tensor(np.zeros((real_HE.size(0)))) # requires_grad = False. Default.
                
                # -----------------------------------------------------------------------------------------
                # Train Generators
                # ------------------------------------------------------------------------------------------
                self.gen_G.train() # train mode
                self.gen_G.train() # train mode
                
                self.gen_optimizer.zero_grad() 
                
                # generate fake_HE and fake_IHC images with the generators
                fake_IHC = self.gen_G(real_HE) 
                fake_IHC = utils.norm_tensor_to_01(fake_IHC)

                fake_HE = self.gen_F(real_IHC)
                fake_HE = utils.norm_tensor_to_01(fake_HE)

                # ssim loss
                ssim_IHC = ssim(fake_IHC, real_IHC)
                ssim_HE = ssim(fake_HE, real_HE)
                loss_ssim = ((1-ssim_IHC) + (1-ssim_HE))/2

                # psnr loss 
                psnr_IHC = psnr(fake_IHC, real_IHC)
                psnr_HE = psnr(fake_HE, real_HE)
                loss_psnr = (psnr_IHC + psnr_HE)/2

                # Identity Loss 
                loss_id_HE = criterion_identity(self.gen_F(real_IHC), real_IHC) 
                loss_id_IHC = criterion_identity(self.gen_G(real_HE), real_HE)
                                                                   
                loss_identity = (loss_id_HE + loss_id_IHC)/2
                
                # GAN Loss
                loss_gen_G = criterion_GAN(fake_IHC, real_IHC) 
                loss_gen_F = criterion_GAN(fake_HE, real_HE) 
                
                loss_GAN = (loss_gen_G +  loss_gen_F)/2
                
                # Cycle Loss
                cycled_IHC = self.gen_G(fake_HE) 
                loss_cycle_G = criterion_cycle(cycled_IHC, real_IHC) 
                cycled_HE = self.gen_F(fake_IHC)
                loss_cycle_F = criterion_cycle(cycled_HE, real_IHC)
                
                loss_cycle = (loss_cycle_G + loss_cycle_F)/2
                
                # apply hyperparameters
                loss_GAN = self.params['generator_lambda']*loss_GAN
                loss_cycle = self.params['cycle_lambda']*loss_cycle
                loss_identity = self.params['identity_lambda']*loss_identity
                loss_ssim = (self.params['ssim_lambda']*loss_ssim)
                loss_psnr = (self.params['psnr_lambda']*loss_psnr)

                # total loss
                loss_G = loss_GAN + loss_cycle + loss_identity + loss_ssim + loss_psnr
                
                loss_G.backward()
                self.gen_optimizer.step()
                
                # ---------------------------------------------------------------------------------
                # Train Discriminator A
                # ---------------------------------------------------------------------------------
                self.disc_X_optimizer.zero_grad()
            
                loss_real = criterion_GAN(self.disc_X(real_HE), valid) # train to discriminate real images as real
                loss_fake = criterion_GAN(self.disc_X(fake_HE.detach()), fake) # train to discriminate fake images as fake
                
                loss_disc_X = (loss_real + loss_fake)/2
                # apply hyperparameter
                loss_disc_Y = self.params['disc_lambda']*loss_disc_Y 
                
                loss_disc_X .backward()
                self.disc_X_optimizer.step()

                # ---------------------------------------------------------------------------------------
                # Train Discriminator B
                # ----------------------------------------------------------------------------------------
                self.disc_Y_optimizer.zero_grad()
            
                loss_real = criterion_GAN(self.disc_Y(real_IHC), valid) # train to discriminate real images as real
                loss_fake = criterion_GAN(self.disc_Y(fake_IHC.detach()), fake) # train to discriminate fake images as fake
                
                loss_disc_Y = (loss_real + loss_fake)/2

                # apply hyperparameter
                loss_disc_Y = self.params['disc_lambda']*loss_disc_Y 
                
                loss_disc_Y .backward()
                self.disc_Y_optimizer.step()
                
                # ------> Total Loss
                loss_D = (loss_disc_X + loss_disc_Y )/2
            
                # -----------------------------------------------------------------------------------------
                # Show Progress
                # --------------------------------------------------------------------------------------------
                if (i+1) % 50 == 0:
                    
                    print('[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [total G loss : %f - (gan : %f, cycle : %f, identity : %f ,ssim : %f ,psnr : %f)]'
                            %(epoch+1,self.params['num_epochs'],       # [Epoch -]
                            i+1,len(train_data_loader),   # [Batch -]
                            loss_D.item(),       # [D loss -]
                            loss_G.item(),       # [total G loss -]
                            loss_GAN.item(),     # [gan -]
                            loss_cycle.item(),   # [cycle -]
                            loss_identity.item(),# [identity -]
                            loss_ssim.item(),# [ssim -]
                            loss_psnr.item(),# [psnr -]
                            ))

        return self.gen_G, self.gen_F, self.disc_X, self.disc_Y



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # reset Conv2d's weight(tensor) with Gaussian Distribution
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) # reset Conv2d's bias(tensor) with Constant(0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # reset BatchNorm2d's weight(tensor) with Gaussian Distribution
            torch.nn.init.constant_(m.bias.data, 0.0) # reset BatchNorm2d's bias(tensor) with Constant(0)

