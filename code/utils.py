import yaml
import matplotlib.pyplot as plt
import numpy as np 
import os
from torchvision import transforms
import torch 
import torch.nn as nn
from pathlib import Path
import models
import loader
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision
import kornia
import models.Simple_U_net.U_net_Generator_model as Simple_U_net
import models.Pix2Pix.U_net_Generator_model as Pix2Pix_UNET
from models.Pix2Pix.Resnet_gen import ResnetGenerator
from models.Pix2Pix.ViT_model import ViT_Generator as pix2pixVit
from models.VisionTransformer.ViT_model import ViT_Generator
from models.DiffusionModel.Diffusion_model import Diffusion
from models.DiffusionModel.Unet_diff import UNet
from models.SwinTransformer.SwinTransformer_model import SwinTransformer

def get_config_from_yaml(config_path):
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.safe_load(stream=param_file)
    return parameters

def save_config_in_dir(saving_dir,code):
    with open(file=saving_dir, mode='w') as fp:
        yaml.dump(code, fp)



def plot_img_set(real_HE, fake_IHC, real_IHC, i,params,img_name,step,epoch):
    if step == 'train' :
        fig_name = str(epoch)+'epoch_Train_plot_'+ img_name[0]+ '.png'
        save_path=os.path.join(os.path.join(params['output_path'],params['output_folder']),"train_plots")
        

    elif step == 'test':
        fig_name = 'Test_plot_'+ img_name[0]+ '.png'
        save_path=os.path.join(os.path.join(params['output_path'],params['output_folder']),"test_plots")

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

    fig.savefig(os.path.join(save_path,fig_name))


def denomalise(mean,std,img):
    
    unorm = transforms.Normalize(mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
                             std=[1/std[0], 1/std[1], 1/std[2]])
    
    denomalised_img = unorm(img)
    return denomalised_img


def hist_loss(self,real_img, fake_img):
    real_img = real_img.squeeze()
    fake_img = fake_img.squeeze()
    hist_loss_list = []
    for c in range(self.params['in_channels']):

        hist_real = torch.histc(real_img[c,:,:], bins=64, min=-5, max=5)
        hist_real  /= hist_real .sum()

        hist_fake = torch.histc(fake_img[c,:,:], bins=64, min=-5, max=5)
        hist_fake /= hist_fake.sum()

        minima = torch.minimum(hist_real, hist_fake)
        intersection = torch.true_divide(minima.sum(), hist_fake.sum())
        hist_loss_list.append(1-intersection)

    return sum(hist_loss_list)




def publishing_plot(images, model_dict):
    # model_architec = {"UNet", "diff_models", "pix2pix", "ViT", "Swin-transfomer"}
    # images["img_num"] = [0,18,199]
    # images["patch_num"] = [0,1,2,3]
    # for each img_num one patch_num
    # len(images["img_num"])==len(images["patch_num"])!!!



    grid = make_grid(img_arr, nrow =len(all_models))
    plot_img = torchvision.transforms.ToPILImage()(grid)
    return plot_img
    




def get_publish_plot_img(images):
    test_path = 'C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/Data_set_BCI_challange/val'

    HE_img_dir = os.path.join(test_path,'HE')
    IHC_img_dir = os.path.join(test_path,'IHC')
    config_path = 'C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/torch_stain_transfer/masterthesis_results/U-Net/4_step_f32/config.yaml'
    # these universal params set image size !
    params = get_config_from_yaml(config_path)

    test_data_0 = loader.stain_transfer_dataset(  img_patch= 0,
                                                        params= params,
                                                        HE_img_dir = HE_img_dir,
                                                        IHC_img_dir = IHC_img_dir,
                                                        )
            
    test_data_loader_0 = DataLoader(test_data_0, batch_size=1, shuffle=False) 

    test_data_1 = loader.stain_transfer_dataset(  img_patch= 1,
                                                        params= params,
                                                        HE_img_dir = HE_img_dir,
                                                        IHC_img_dir = IHC_img_dir,
                                                        )
            
    test_data_loader_1 = DataLoader(test_data_0, batch_size=1, shuffle=False) 

    test_data_2 = loader.stain_transfer_dataset(  img_patch= 2,
                                                        params= params,
                                                        HE_img_dir = HE_img_dir,
                                                        IHC_img_dir = IHC_img_dir,
                                                        )
            
    test_data_loader_2 = DataLoader(test_data_0, batch_size=2, shuffle=False) 

    test_data_3 = loader.stain_transfer_dataset(  img_patch= 3,
                                                        params= params,
                                                        HE_img_dir = HE_img_dir,
                                                        IHC_img_dir = IHC_img_dir,
                                                        )
            
    test_data_loader_3 = DataLoader(test_data_3, batch_size=1, shuffle=False) 



    all_img =np.array(images["img_num"])
    all_img_patches =np.array(images["patch_num"])

    images_0_patch = all_img[np.where(all_img_patches==0)]     
    images_1_patch = all_img[np.where(all_img_patches==1)] 
    images_2_patch = all_img[np.where(all_img_patches==2)] 
    images_3_patch = all_img[np.where(all_img_patches==3)]  

    plot_img_IHC = []
    plot_img_HE = []
 
    for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader_0):
        if i in images_0_patch:
            plot_img_IHC.append(real_IHC)
            plot_img_HE.append(real_HE)


    for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader_1):
        if i in images_1_patch:
            plot_img_IHC.append(real_IHC)
            plot_img_HE.append(real_HE)



    for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader_2):
        if i in images_1_patch:
            plot_img_IHC.append(real_IHC)
            plot_img_HE.append(real_HE)


    for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader_3):
        if i in images_1_patch:
            plot_img_IHC.append(real_IHC)
            plot_img_HE.append(real_HE)

    return plot_img_IHC, plot_img_HE






def load_trained_model(architecture_name,version):
        # ----------------U-Net---------------------------------
    if architecture_name == "U-Net" :
            
        path = 'masterthesis_results\\U-Net\\'+version
        config_path = os.path.join(Path.cwd(),path+'\\config.yaml')
        model_path = os.path.join(path,'gen_G_weights_final.pth')
        params = get_config_from_yaml(config_path)

        UNet_model = Simple_U_net.U_net_Generator(in_channels=params['in_channels'], 
                                                                      out_channels=3, 
                                                                      features=params['num_features'], 
                                                                      steps=params['num_steps'], 
                                                                      bottleneck_len=params['bottleneck_len']).to(params['device'])
        UNet_model.load_state_dict(torch.load(model_path))
        model_name =['U-Net---'+version]

        return UNet_model , model_name
        
        # ----------------------------- pix2pix -------------------------------------------
    if architecture_name == "pix2pix":
            
        path = 'masterthesis_results\\pix2pix\\'+version
        config_path = os.path.join(Path.cwd(),path+'\\config.yaml')
        model_path = os.path.join(path,'gen_G_weights_final.pth')
        params = get_config_from_yaml(config_path)

        if params['gen_architecture']== "my_Unet":

            pix2pix_model = Pix2Pix_UNET.U_net_Generator(in_channels=params['in_channels'], 
                                                                    out_channels=3, 
                                                                    features=params['num_features'], 
                                                                    steps=params['num_steps'], 
                                                                    bottleneck_len=params['bottleneck_len']).to(params['device'])
                    
        if params['gen_architecture']== "Resnet":
            pix2pix_model = ResnetGenerator(input_nc=params['in_channels'], output_nc=3, ngf=params['num_features'], n_blocks=9).to(params['device'])

        if params['gen_architecture']== "transformer":
            pix2pix_model = pix2pixVit(   chw = [params['in_channels']]+params['img_size'], 
                                                    patch_size = params['patch_size'],
                                                    num_heads = params['num_heads'], 
                                                    num_blocks = params['num_blocks'],
                                                    attention_dropout = params['attention_dropout'], 
                                                    dropout= params['dropout'],
                                                    mlp_ratio=params['mlp_ratio']
                                                    ).to(params['device'])
                    

                
        pix2pix_model.load_state_dict(torch.load(model_path))
        model_name =['pix2pix---'+version]

        return pix2pix_model , model_name
        
        # --------------------------- ViT --------------------------------------
    if architecture_name == "ViT":
            
        path = 'masterthesis_results\\ViT\\'+version
        config_path = os.path.join(Path.cwd(),path+'\\config.yaml')
        model_path = os.path.join(path,'gen_G_weights_final.pth')
        params = get_config_from_yaml(config_path)

        ViT_model = ViT_Generator(  chw = [params['in_channels']]+params['img_size'], 
                                                                                patch_size = params['patch_size'],
                                                                                num_heads = params['num_heads'], 
                                                                                num_blocks = params['num_blocks'],
                                                                                attention_dropout = params['attention_dropout'], 
                                                                                dropout= params['dropout'],
                                                                                mlp_ratio=params['mlp_ratio']
                                                                                ).to(params['device'])
        ViT_model.load_state_dict(torch.load(model_path))
        model_name =['ViT---'+version]

        return ViT_model, model_name
        

    if architecture_name == "diff_model":
        path = 'masterthesis_results\\diffusion_model\\'+version
        config_path = os.path.join(Path.cwd(),path+'\\config.yaml')
        model_path = os.path.join(path,'gen_G_weights_final.pth')
        params = get_config_from_yaml(config_path)
        diff_model = UNet().to(params['device'])
        diff_model.load_state_dict(torch.load(model_path))
                
        diffusion = Diffusion(noise_steps=params['noise_steps'],
                              beta_start=params['beta_start'],
                              beta_end=params['beta_end'],
                              img_size=params['img_size'],
                              device=params['device'])
                
        diff_model.load_state_dict(torch.load(model_path))
        model_name =['diffusion---'+version]

        return diffusion, diff_model, model_name
    
    if architecture_name == "Swin_transformer":
            
        path = 'masterthesis_results\\Swin_transformer\\'+version
        config_path = os.path.join(Path.cwd(),path+'\\config.yaml')
        model_path = os.path.join(path,'gen_G_weights_final.pth')
        params = get_config_from_yaml(config_path)

        Swin_model = SwinTransformer( hidden_dim=params['hidden_dim'], 
                                                                                        layers=params['layers'], 
                                                                                        heads=params['heads'], 
                                                                                        in_channels=params['in_channels'], 
                                                                                        out_channels=params['out_channels'], 
                                                                                        head_dim=params['head_dim'], 
                                                                                        window_size=params['window_size'],
                                                                                        downscaling_factors=params['downscaling_factors'], 
                                                                                        relative_pos_embedding=params['relative_pos_embedding']
                                                                                        ).to(params['device'])
        Swin_model.load_state_dict(torch.load(model_path))
                
        ViT_model.load_state_dict(torch.load(model_path))
        model_name =['Swin_transformer---'+version]
        return Swin_model , model_name



 
                
        
            
       
        

        
