import utils
import numpy as np 
import matplotlib.pyplot as plt
import torch
from models.DiffusionModel.Diffusion_model import Diffusion
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

model_architectures = {}
#model_architectures["U-Net"] = ['my_UNet','my_UNet_color_loss_only']
#model_architectures["diffusion_model"] = ['diff_1000t_loaded']
#model_architectures["pix2pix"] = ['my_UNet_color_loss_only','Transformer']
model_architectures["ViT"] = ['Vit_1Block_4Mlp_4pat_mse+gaus']
model_architectures["swin_transformer"] = ['2_stage_win4_96_dim_+gaus']
column_labels = ['HE \n Input','IHC \n Target','ViT','Swin_tr']
row_labels = ['img_1', 'img_2', 'img_3', 'img_4']
images = {}
images["img_num"]=[2,4,65,45]
images["patch_num"]= [0,0,0,0]

#grid, columb_names = utils.publishing_plot(images, model_architectures)




def publishing_plot(images, model_dict,column_labels, row_labels):
    # model_architec = {"UNet", "diff_models", "pix2pix", "ViT", "Swin-transfomer"}
    # images["img_num"] = [0,18,199]
    # images["patch_num"] = [0,1,2,3]
    # for each img_num one patch_num
    # len(images["img_num"])==len(images["patch_num"])!!!
    plot_img_IHC, plot_img_HE = utils.get_publish_plot_img(images=images)

    img_arr = []
    num_samples = len(plot_img_IHC)

    
    for idx in range(len(plot_img_IHC)):
        real_HE = plot_img_HE[idx]
        real_IHC = plot_img_IHC[idx]

        real_HE_plot = real_HE.cpu().detach().numpy()
        real_IHC_plot = real_IHC.cpu().detach().numpy()

        real_HE_plot = np.squeeze(real_HE_plot )
        real_IHC_plot = np.squeeze(real_IHC_plot )

        real_IHC_plot = np.transpose(real_IHC_plot, (1, 2, 0))
        real_HE_plot  = np.transpose(real_HE_plot , (1, 2, 0))

        #real_HE_plot = torch.from_numpy(real_HE_plot) 
        #real_IHC_plot = torch.from_numpy(real_IHC_plot)

        img_arr.append(real_HE_plot)
        img_arr.append(real_IHC_plot)


        for architecture_name in model_dict:
            for version in model_dict[architecture_name]:

                model, model_name, params = utils.load_trained_model(architecture_name,version)
                
            

                
                if model_name[0].__contains__('diffusion---'):
                        
                    diffusion = Diffusion(noise_steps=params['noise_steps'],
                                            beta_start=params['beta_start'],
                                            beta_end=params['beta_end'],
                                            img_size=params['img_size'],
                                            device=params['device'])
                    fake_IHC = diffusion.sample(model , n=real_IHC.shape[0],y=real_HE)
                else:
                    fake_IHC = model(real_HE)

                fake_IHC = fake_IHC.cpu().detach().numpy()
                fake_IHC = np.squeeze(fake_IHC)
                fake_IHC = np.transpose(fake_IHC, (1, 2, 0))

                #fake_IHC = torch.from_numpy(fake_IHC)
                img_arr.append(fake_IHC)






    images = img_arr  
    num_rows = num_samples  
    num_cols = len(column_labels) 

    #fig, axes = plt.subplots(num_rows, num_cols,figsize=(num_rows+4, num_cols+4))
    #fig.set_aspect('equal')

    # Set the size of each subplot
    subplot_size = 3  # Adjust this value to control the size of each subplot
    fig_width = subplot_size * num_cols
    fig_height = subplot_size * num_rows + 1.0 
    # Create a figure with a size that accommodates the subplots and labels
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    for ax in axes.ravel():
        ax.set_aspect('equal')
    
    
    # Create subplots and labels
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < len(images):
                axes[i, j].imshow(images[index])
                axes[i, j].get_xaxis().set_visible(False)
                axes[i, j].get_yaxis().set_visible(False)
                #axes[i, j].axis('off')

    # Labels for columns at the very top
    
    for j, label in enumerate(column_labels):
        ax = axes[0, j]
        ax.set_title(label, fontsize=15, pad=10)  # Use pad to control vertical spacing
 

    


    plt.subplots_adjust(wspace=0, hspace=0.01)
    plt.savefig('foo.png')
    return fig

figure = publishing_plot(images, model_architectures,column_labels, row_labels)
    




