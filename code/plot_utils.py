import matplotlib.pyplot as plt
import os
import numpy as np
import new_loader
from torch.utils.data import DataLoader
from pathlib import Path
from architectures.Diffusion_model import Diffusion
import torch
import utils



# plot the train MSE/SSIM/PSNR
def plot_trainresult(args,save_path,train_eval):
    x = range(len(train_eval['train_mse']))
    fig, axs = plt.subplots(3)
    fig.suptitle('Training and Validation results')

    axs[0].plot(x, train_eval['train_mse'],label='train_mse')
    if not args.diff_model:
        axs[0].plot(x, train_eval['val_mse'],label='val_mse')
    axs[0].legend(loc="upper right",fontsize='xx-small')
    axs[0].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
    axs[0].set_title('MSE',loc='left')

    axs[1].plot(x, train_eval['train_ssim'],label='train_ssim')
    if not args.diff_model:
        axs[1].plot(x, train_eval['val_ssim'],label='val_ssim')
    axs[1].legend(loc="lower right",fontsize='xx-small')
    axs[1].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
    axs[1].set_title('SSIM',loc='left')

    axs[2].plot(x, train_eval['train_psnr'],label='train_psnr')
    if not args.diff_model:
        axs[2].plot(x, train_eval['val_psnr'],label='val_psnr')
    axs[2].legend(loc="lower right",fontsize='xx-small')
    axs[2].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
    axs[2].set_title('PSNR',loc='left')
        
    plt.subplots_adjust(hspace=1.3)
    fig.savefig(os.path.join(save_path,"Train_loss_plots.png"))


# plot image set in the train and test loop 
def plot_img_set(real_HE, fake_IHC, real_IHC, save_path,img_name,epoch):
    
    fig_name = str(epoch)+'_epoch_'+ img_name[0]
        
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

# get all images for publisching plot 
def get_publish_plot_img(args,images):
    patches = np.unique(images['patch_num'])
    all_img =np.array(images["img_num"])
    all_img_patches =np.array(images["patch_num"])

    plot_img_IHC = []
    plot_img_HE = []

    for current_patch in patches:

        test_data = new_loader.stain_transfer_dataset( img_patch=current_patch, set='train',args = args) 
        test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

    


        images_patch = all_img[np.where(all_img_patches==current_patch)]     
 


        for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader):
            if i in images_patch:
                plot_img_IHC.append(real_IHC)
                plot_img_HE.append(real_HE)


    return plot_img_IHC, plot_img_HE

# predict all images with models for publishing plot
def predict_all_img(args,images,model_dict):
    plot_img_IHC, plot_img_HE = get_publish_plot_img(images=images)

    img_arr = []
    num_samples = len(plot_img_IHC)

    # cycle through images
    for idx in range(len(plot_img_IHC)):
        real_HE = plot_img_HE[idx]
        real_IHC = plot_img_IHC[idx]

        real_HE_plot = real_HE.cpu().detach().numpy()
        real_IHC_plot = real_IHC.cpu().detach().numpy()

        real_HE_plot = np.squeeze(real_HE_plot )
        real_IHC_plot = np.squeeze(real_IHC_plot )

        real_IHC_plot = np.transpose(real_IHC_plot, (1, 2, 0))
        real_HE_plot  = np.transpose(real_HE_plot , (1, 2, 0))

        img_arr.append(real_HE_plot)
        img_arr.append(real_IHC_plot)

        #cycle through models 
        num_models = 0
        for architecture_name in model_dict:
            for version in model_dict[architecture_name]:
                args.model = architecture_name
                args.type = version
                
                model, model_name = utils.load_model(args)
                model_dir = os.path.join(Path.cwd(),"masterthesis_results")
                train_path = os.path.join(model_dir,model_name)
                if os.path.isdir(train_path):
                    best_model_weights = os.path.join(train_path,'final_weights_gen.pth')
                    model.load_state_dict(torch.load(best_model_weights))
                else:
                    print('MODEL NOT TRAINED CHECK MODEL DICT')

                
                if model_name[0].__contains__('diff'):
                        
                    diffusion = Diffusion(noise_steps=args.diff_noise_steps,img_size=args.img_size,device=args.device) 
                    fake_IHC = diffusion.sample(model , n=real_IHC.shape[0],y=real_HE)
                else:
                    fake_IHC = model(real_HE)

                fake_IHC = fake_IHC.cpu().detach().numpy()
                fake_IHC = np.squeeze(fake_IHC)
                fake_IHC = np.transpose(fake_IHC, (1, 2, 0))

                #fake_IHC = torch.from_numpy(fake_IHC)
                img_arr.append(fake_IHC)
                num_models = num_models+1
    return img_arr, num_samples, num_models

def get_publishing_plot(args,images,model_dict,save_path,plot_name):
    img_arr, num_samples, num_models = predict_all_img(args, images, model_dict)
    images = img_arr  
    num_rows = num_samples  
    num_cols = num_models
    column_labels = ['HE \n Input','IHC \n Target']
    for architecture_name in model_dict:
            for version in model_dict[architecture_name]:
                column_labels.append(architecture_name+ '\n'+version)


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
                

    # Labels for columns at the very top
    for j, label in enumerate(column_labels):
        ax = axes[0, j]
        ax.set_title(label, fontsize=15, pad=10)  
 

    


    plt.subplots_adjust(wspace=0, hspace=0.01)
    plt.savefig(os.path.join(save_path,plot_name))
    return fig