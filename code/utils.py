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




def publishing_plot(images, model_names= ["UNet", "diff_model", "pix2pix", "ViT", "Swin_transfomer"]):
    # model_names = ["UNet", "diff_models", "pix2pix", "ViT", "Swin-transfomer"]
    # images["names"]
    # images["patch_num"]

    test_path = 'C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/Data_set_BCI_challange/val'

    HE_img_dir = os.path.join(test_path,'HE')
    IHC_img_dir = os.path.join(test_path,'IHC')
    config_path = 'C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/torch_stain_transfer/masterthesis_results/U-Net/4_step_f32/config.yaml'
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

    all_models = []

    
    for  model_name in model_names:
        
        if model_name == "UNet":
            config_path = os.path.join(Path.cwd(),'masterthesis_results\\U-Net\\4-step_f32\\config.yaml')
            output_folder_path = os.path.join(params['output_path'],params['output_folder'])
            model_path = os.path.join(output_folder_path,params['model_name'])
            params = get_config_from_yaml(config_path)
            UNet_model = models.Simple_U_net.U_net_model.UNet(in_channels=params['in_channels'],out_channels=3, init_features=params['gen_features']).to(params['device'])
            UNet_model.load_state_dict(torch.load(model_path))
            all_models.append(UNet_model)
            

        if model_name == "pix2pix":
            config_path = os.path.join(Path.cwd(),'masterthesis_results\\pix2pix\\gen_f32_disc_f32\\config.yaml')
            output_folder_path = os.path.join(params['output_path'],params['output_folder'])
            model_path = os.path.join(output_folder_path,params['model_name'])
            params = get_config_from_yaml(config_path)
            pix2pix_model = models.Pix2Pix.U_net_model.UNet(in_channels=params['in_channels'],out_channels=3, init_features=params['gen_features']).to(params['device'])
            pix2pix_model.load_state_dict(torch.load(model_path))
            all_models.append(pix2pix_model)
            

        if model_name == "ViT":
            config_path = os.path.join(Path.cwd(),'masterthesis_results\\ViT\\16_patch\\config.yaml')
            output_folder_path = os.path.join(params['output_path'],params['output_folder'])
            model_path = os.path.join(output_folder_path,params['model_name'])
            params = get_config_from_yaml(config_path)
            ViT_model = models.VisionTransformer.ViT_model.ViT_Generator(  chw = [params['in_channels']]+params['img_size'], 
                                                                                patch_size = params['patch_size'],
                                                                                num_heads = params['num_heads'], 
                                                                                num_blocks = params['num_blocks'],
                                                                                attention_dropout = params['attention_dropout'], 
                                                                                dropout= params['dropout'],
                                                                                mlp_ratio=params['mlp_ratio']
                                                                                ).to(params['device'])
            ViT_model.load_state_dict(torch.load(model_path))
            all_models.append(ViT_model)
            

        if model_name == "diff_model":
            config_path = os.path.join(Path.cwd(),'masterthesis_results\\diffusion_model\\diff_1000t\\config.yaml')
            output_folder_path = os.path.join(params['output_path'],params['output_folder'])
            model_path = os.path.join(output_folder_path,params['model_name'])
            params = get_config_from_yaml(config_path)
            diff_model = models.DiffusionModel.Unet_diff.UNet().to(params['device'])
            diff_model.load_state_dict(torch.load(model_path))
            all_models.append(diff_model)
            diffusion = models.DiffusionModelDiffusion(noise_steps=params['noise_steps'],beta_start=params['beta_start'],beta_end=params['beta_end'],img_size=params['img_size'],device=params['device'])
            


        if model_name == "Swin_transformer":
            config_path = os.path.join(Path.cwd(),'masterthesis_results\\Swin_transformer\\win_8\\config.yaml')
            output_folder_path = os.path.join(params['output_path'],params['output_folder'])
            model_path = os.path.join(output_folder_path,params['model_name'])
            params = get_config_from_yaml(config_path)
            Swin_model = models.SwinTransformer.SwinTransformer_model.SwinTransformer( hidden_dim=params['hidden_dim'], 
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
            all_models.append(Swin_model)
            
    c = 0
    for  image_name_plot in images["names"]:


        if images["patch_num"][c]==0:
            img_arr = []
            for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader_0):
                if img_name == image_name_plot:
                    for gen_model in all_models:
                        if gen_model == diff_model:
                            fake_IHC = diffusion.sample(gen_model , n=real_IHC.shape[0],y=real_HE)
                        else:
                            fake_IHC = gen_model(real_HE)
                        img_arr.append(fake_IHC)


        elif images["patch_num"][c]==1:
            for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader_1):
                if img_name == image_name_plot:
                    for gen_model in all_models:
                        if gen_model == diff_model:
                            fake_IHC = diffusion.sample(gen_model , n=real_IHC.shape[0],y=real_HE)
                        else:
                            fake_IHC = gen_model(real_HE)
                        img_arr.append(fake_IHC)


        elif images["patch_num"][c]==2:
            for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader_2):
                if img_name == image_name_plot:
                    for gen_model in all_models:
                        if gen_model == diff_model:
                            fake_IHC = diffusion.sample(gen_model , n=real_IHC.shape[0],y=real_HE)
                        else:
                            fake_IHC = gen_model(real_HE)
                        img_arr.append(fake_IHC)


        elif images["patch_num"][c]==3:
            for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader_3):
                if img_name == image_name_plot:
                    for gen_model in all_models:
                        if gen_model == diff_model:
                            fake_IHC = diffusion.sample(gen_model , n=real_IHC.shape[0],y=real_HE)
                        else:
                            fake_IHC = gen_model(real_HE)
                        img_arr.append(fake_IHC)
        c =c+1
        grid = make_grid(img_arr, nrow =len(all_models))
        plot_img = torchvision.transforms.ToPILImage()(grid)
        return plot_img
    
def frequency_division(src_img):
        #input:src_img    type:tensor
        #output:image_low,image_high    type:tensor
        #get low frequency component and high frequency compinent of image
        fft_src = torch.fft.rfft( src_img, dim=2)
        fft_amp = (fft_src[:,:,:,:,0]**2) + (fft_src[:,:,:,:,1]**2)
        fft_amp = torch.sqrt(fft_amp)
        fft_pha = torch.atan2( fft_src[:,:,:,:,1], fft_src[:,:,:,:,0] )

        # replace the low frequency amplitude part of source with that from target
        _, _, h, w = fft_amp.size()
        amp_low = torch.zeros(fft_amp.size(), dtype=torch.float)
        b = (  np.floor(np.amin((h,w))*0.1)  ).astype(int)     # get b
        amp_low[:,:,0:b,0:b]     = fft_amp[:,:,0:b,0:b]      # top left
        amp_low[:,:,0:b,w-b:w]   = fft_amp[:,:,0:b,w-b:w]    # top right
        amp_low[:,:,h-b:h,0:b]   = fft_amp[:,:,h-b:h,0:b]    # bottom left
        amp_low[:,:,h-b:h,w-b:w] = fft_amp[:,:,h-b:h,w-b:w]  # bottom right
        amp_high = fft_amp - amp_low
        # recompose fft of source
        fft_low = torch.zeros( fft_src.size(), dtype=torch.float )
        fft_high = torch.zeros( fft_src.size(), dtype=torch.float )
        fft_low[:,:,:,:,0] = torch.cos(fft_pha) * amp_low
        fft_low[:,:,:,:,1] = torch.sin(fft_pha) * amp_low
        fft_high[:,:,:,:,0] = torch.cos(fft_pha)* amp_high
        fft_high[:,:,:,:,1] = torch.sin(fft_pha)* amp_high

        # get the recomposed image: source content, target style
        _, _, imgH, imgW = src_img.size()
        image_low = torch.fft.irfft(fft_low, dim=2)
        image_high = torch.fft.irfft(fft_high, im=2)
        return image_low,image_high

