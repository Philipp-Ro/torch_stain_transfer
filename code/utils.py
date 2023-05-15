import yaml
import matplotlib.pyplot as plt
import numpy as np 
import os
from torchvision import transforms
import torch 


def get_config_from_yaml(config_path):
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.safe_load(stream=param_file)
    return parameters

def save_config_in_dir(saving_dir,code):
    with open(file=saving_dir, mode='w') as fp:
        yaml.dump(code, fp)



def plot_img_set(real_HE, fake_IHC, real_IHC, i,params,img_name):
    fig_name = 'plot_'+ img_name[0]+ '.png'

    real_HE = real_HE.cpu().detach().numpy()
    fake_IHC = fake_IHC.cpu().detach().numpy()
    real_IHC = real_IHC.cpu().detach().numpy()

    real_HE = np.squeeze(real_HE )
    fake_IHC = np.squeeze(fake_IHC)
    real_IHC = np.squeeze(real_IHC )

    real_HE = np.transpose(real_HE, axes=[1,2,0])
    fake_IHC = np.transpose(fake_IHC, axes=[1,2,0])
    real_IHC = np.transpose(real_IHC, axes=[1,2,0])
    

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)       
    plt.imshow(real_HE )
    plt.axis('off')
    plt.title('real_HE')


    fig.add_subplot(1, 3, 2)       
    plt.imshow(fake_IHC )
    plt.axis('off')
    plt.title('fake_IHC')
            
    fig.add_subplot(1, 3, 3)    
    plt.imshow(real_IHC )
    plt.axis('off')
    plt.title('real_IHC')

    fig.savefig(os.path.join(os.path.join(params['output_path'],params['output_folder']),fig_name))


def denomalise(mean,std,img):
    unorm = transforms.Normalize(mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
                             std=[1/std[0], 1/std[1], 1/std[2]])
    
    denomalised_img = unorm(img)
    return denomalised_img

## gradient penalty for wasserstein Gan with gradient penalty 
# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/4.%20WGAN-GP
def gradient_penalty(critic, real, fake):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).cuda()
    interpolated_images = real * alpha + fake * (1 - alpha)
    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# ---------------------------------------------------------------------------------
# Discriminator training function 
# # ---------------------------------------------------------------------------------

def discriminator_loss(self, disc, real_img, fake_img, params):
    # inputs:
    # disc -------------> initialized Discriminator model
    # real_img ---------> real HE or IHC img 
    # fake_img ---------> fake HE or IHC img 
    # params -----------> training parameters 
                  
    if 'gan_loss' in params['total_loss_comp']:

        # ------------------ train to discriminate fake images as fake --------------------------------
        # detach() generated image so that the grpah doesnt go through !!!! 
        disc_pred_fake = disc(fake_img.detach()).flatten()
        disc_probablity_fake = self.sigmoid(disc_pred_fake)

        loss_fake = self.criterion_GAN(disc_probablity_fake, self.fake) 
        loss_fake_scaled = loss_fake*params['disc_lambda']

        # ------------------ train to discriminate real images as real --------------------------------
        disc_pred_real = disc(real_img).flatten()
        disc_probablity_real = self.sigmoid(disc_pred_real)

        loss_real = self.criterion_GAN(disc_probablity_real, self.valid) 
        loss_real_scaled = loss_real *params['disc_lambda']
     
        # ------------------ combine losses for total loss ---------------------------------------------
        loss_total = (loss_real_scaled+loss_fake_scaled)/2
     

                    
    elif'wgan_loss'in params['total_loss_comp']:
        # https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490

        loss_critic  = -(torch.mean(disc(fake_img.detach())) - torch.mean(self.disc(real_img)))
        loss_total = loss_critic

    return loss_total


        
# ---------------------------------------------------------------------------------
# Generators training function 
# ---------------------------------------------------------------------------------

def generator_loss(self, disc, fake_img, params):
    # inputs:
    # disc -------------> initialized Discriminator model
    # fake_img ---------> fake HE or IHC img 
    # params -----------> training parameters 
    if 'gan_loss' in params['total_loss_comp']:
        disc_pred_fake = disc(fake_img.detach()).flatten()
        disc_probablity_fake = self.sigmoid(disc_pred_fake)

        loss_gen = self.criterion_GAN(disc_probablity_fake, self.valid) 
        loss_gen = self.params['generator_lambda']*loss_gen



    elif'wgan_loss'in self.params['total_loss_comp']:
        loss_gen= -1. * torch.mean(self.disc(fake_img.detach()))

    else :
        print('CHOOSE gan_loss OR wgan_loss  IN total_loss_comp IN THE YAML FILE' )
 
    return loss_gen
