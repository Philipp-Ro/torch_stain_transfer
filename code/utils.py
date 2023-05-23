import yaml
import matplotlib.pyplot as plt
import numpy as np 
import os
from torchvision import transforms
import torch 
import torch.nn as nn

def get_config_from_yaml(config_path):
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.safe_load(stream=param_file)
    return parameters

def save_config_in_dir(saving_dir,code):
    with open(file=saving_dir, mode='w') as fp:
        yaml.dump(code, fp)



def plot_img_set(real_HE, fake_IHC, real_IHC, i,params,img_name,step,epoch):
    if step == 'train' :
        fig_name = 'Train_plot_'+ img_name[0]+ str(epoch)+'.png'

    if step == 'test':
        fig_name = 'Test_plot_'+ img_name[0]+ '.png'
    else:
        print('SET STEP TO TRAIN OR TEST ')

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
    label_real = torch.cuda.FloatTensor(np.ones((real_img.size(0))))
    label_fake = torch.cuda.FloatTensor(np.zeros((fake_img.size(0)))) 

    if 'gan_loss' in params['total_loss_comp']:

        # ------------------ train to discriminator --------------------------------
        # detach() generated image so that the grpah doesnt go through !!!! 
        # sigmoid layer for which models as last layer ? 
        # disc_pred_fake = disc(fake_img.detach()).flatten()
        # disc_probablity_fake = self.sigmoid(disc_pred_fake)
        loss_total = 0

        disc_pred_fake = disc(fake_img).flatten()
        disc_pred_real = disc(real_img).flatten()

        fake_validity = nn.Sigmoid()(disc_pred_fake.view(-1))
        real_validity = nn.Sigmoid()(disc_pred_real.view(-1))

        d_fake_loss = nn.BCELoss()(fake_validity, label_fake)
        d_real_loss = nn.BCELoss()(real_validity, label_real)

        loss_fake_scaled = d_fake_loss*params['disc_lambda']
        loss_real_scaled = d_real_loss *params['disc_lambda']
     
        # ------------------ combine losses for total loss ---------------------------------------------
        loss_total = (loss_real_scaled+loss_fake_scaled)/2

    elif 'MSE_loss'in params['total_loss_comp']:

        loss_total = 0

        disc_pred_fake = disc(fake_img).flatten()
        disc_pred_real = disc(real_img).flatten()

        d_fake_loss = nn.MSELoss()(disc_pred_fake, label_fake)
        d_real_loss = nn.MSELoss()(disc_pred_real, label_real)


        loss_total += d_real_loss + d_fake_loss
                    
    elif'wgan_loss_gp'in params['total_loss_comp'] or 'wgan_loss' in params['total_loss_comp']:
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
    label_real = torch.cuda.FloatTensor(np.ones((real_img.size(0))))
    label_fake = torch.cuda.FloatTensor(np.zeros((fake_img.size(0)))) 
    if 'gan_loss' in params['total_loss_comp']:
        loss_gen = 0
        # sigmoid layer for which models as last layer ? 
        disc_pred_fake = disc(fake_img).flatten()
        fake_validity = nn.Sigmoid()(disc_pred_fake.view(-1))

        loss_gen = nn.BCELoss()(fake_validity.view(-1), label_real)

        loss_gen = self.params['generator_lambda']*loss_gen

    elif 'MSE_loss'in params['total_loss_comp']:
        loss_gen = 0
        disc_pred_fake = disc(fake_img).flatten()
        loss_gen = nn.MSELoss()(disc_pred_fake, label_real)
        loss_gen = self.params['generator_lambda']*loss_gen

    elif'wgan_loss_gp'in self.params['total_loss_comp'] or 'wgan_loss'in self.params['total_loss_comp'] :
        loss_gen = 0
        loss_gen=  torch.mean(self.disc(fake_img.detach()))
        loss_gen = self.params['generator_lambda']*loss_gen
    else :
        print('CHOOSE gan_loss OR wgan_loss OR wgan_loss_gp  IN total_loss_comp IN THE YAML FILE' )
 
    return loss_gen
