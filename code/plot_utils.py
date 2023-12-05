import matplotlib.pyplot as plt
import os
import numpy as np
import new_loader
from torch.utils.data import DataLoader
from pathlib import Path
from architectures.Diffusion_model import Diffusion
import torch
import utils
import torchvision.transforms as T



# plot the train MSE/SSIM/PSNR
def plot_trainresult(args, save_path, train_eval, test_eval):
    fig, axs = plt.subplots(3)

    axs[0].plot(train_eval['x'], train_eval['MSE'],label='train MSE')
    if not args.diff_model:
        axs[0].plot(test_eval['x'], test_eval['MSE'],label='test MSE')
    axs[0].legend(loc="upper right",fontsize='xx-small')
    axs[0].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
    axs[0].set_title('MSE',loc='left')

    axs[1].plot(train_eval['x'], train_eval['SSIM'],label='train SSIM')
    if not args.diff_model:
        axs[1].plot(test_eval['x'], test_eval['SSIM'],label='test SSIM')
    axs[1].legend(loc="lower right",fontsize='xx-small')
    axs[1].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
    axs[1].set_title('SSIM',loc='left')

    axs[2].plot(train_eval['x'], train_eval['PSNR'],label='train PSNR')
    if not args.diff_model:
        axs[2].plot(test_eval['x'], test_eval['PSNR'],label='test PSNR')
    axs[2].legend(loc="lower right",fontsize='xx-small')
    axs[2].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
    axs[2].set_title('PSNR',loc='left')
        
    plt.subplots_adjust(hspace=1.3)
    figurename = "Train_loss_plots_.png"
    fig.savefig(os.path.join(save_path,figurename))


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


def get_imgs(args, img_names, model):
    img_arr = []

    args.img_size = 1024
    transform_resize = T.Resize((256,256))

    if args.model == 'Diffusion':
        diffusion = Diffusion(noise_steps=args.diff_noise_steps,img_size=args.img_size,device=args.device)

    test_data = new_loader.stain_transfer_dataset( img_patch=0, set='test',args = args) 
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

    for i, (real_HE, real_IHC,img_name) in enumerate(test_data_loader) :
                 
        real_HE = transform_resize(real_HE)
        real_IHC = transform_resize(real_IHC)
 
        if img_name[0] in img_names :

            if args.model == 'Diffusion':
                fake_IHC = diffusion.sample(model , n=real_IHC.shape[0], y=real_HE)

            elif args.model == 'source':
                fake_IHC = []

                real_HE_plot = real_HE.cpu().detach().numpy()
                real_HE_plot = np.squeeze(real_HE_plot )
                real_IHC_plot = np.transpose(real_IHC_plot, (1, 2, 0))
                img_arr.append(real_HE_plot)

            elif args.model == 'target':
                fake_IHC = []

                real_IHC_plot = real_IHC.cpu().detach().numpy()
                real_IHC_plot = np.squeeze(real_IHC_plot )
                real_IHC_plot = np.transpose(real_IHC_plot, (1, 2, 0))
                img_arr.append(real_IHC_plot)
                

            else:
                fake_IHC = model(real_HE)

                fake_IHC = fake_IHC.cpu().detach().numpy()
                fake_IHC = np.squeeze(fake_IHC)
                fake_IHC = np.transpose(fake_IHC, (1, 2, 0))
                img_arr.append(fake_IHC)

        return img_arr
    
def get_imgs_for_all_models(args, model_dict, img_names):
    img_arr = []
    model = []
    model_labels = []
    # get HE imgs
    args.model = 'source'
    img_arr_source = get_imgs(args, img_names, model)
    img_arr = np.concatenate((img_arr, img_arr_source), axis=0)
    

    # get IHC imgs
    args.model = 'target'
    img_arr_target = get_imgs(args, img_names, model)
    img_arr = np.concatenate((img_arr, img_arr_target), axis=0)

    # get images from all networks in model_dict
    for architecture_name in model_dict:
        args.model = architecture_name
  
        for version in model_dict[architecture_name]:
            args.type = version
            model, model_name = utils.load_model(args)
            model = utils.load_model_weights(model, model_name)
            model = model.to(args.device)

            img_arr_model = get_imgs(args, img_names, model)
            img_arr = np.concatenate((img_arr, img_arr_model), axis=0)

            model_label_name = args.model +'\n'+args.type
            model_labels.append(model_label_name)

    return img_arr, model_labels

def save_plot_for_models(args, model_dict, IHC_score):

    img_arr, model_labels = get_imgs_for_all_models(args, model_dict, img_names)

    num_rows = len(model_labels) +2
    num_cols = len(img_names)

    if IHC_score == '0':
        img_names = []
        plot_name = 'all_models_IHC_score_0'
        column_labels = ['img 1\nIHC '+IHC_score,'img 2\nIHC '+IHC_score,'img 3\nIHC '+IHC_score,'img 4\nIHC '+IHC_score]

    if IHC_score == '1+':
        img_names = []
        plot_name = 'all_models_IHC_score_1+'
        column_labels = ['img 1\nIHC '+IHC_score,'img 2\nIHC '+IHC_score,'img 3\nIHC '+IHC_score,'img 4\nIHC '+IHC_score]

    if IHC_score == '2+':
        img_names = []
        plot_name = 'all_models_IHC_score_2+'
        column_labels = ['img 1\nIHC '+IHC_score,'img 2\nIHC '+IHC_score,'img 3\nIHC '+IHC_score,'img 4\nIHC '+IHC_score]

    if IHC_score == '3+':
        img_names = []
        plot_name = 'all_models_IHC_score_3+'
        column_labels = ['img 1\nIHC '+IHC_score,'img 2\nIHC '+IHC_score,'img 3\nIHC '+IHC_score,'img 4\nIHC '+IHC_score]

    if IHC_score == 'all':
        img_names = ['01269_train_0.png','00864_train_1+.png','00265_train_2+.png','00156_train_3+.png']
        plot_name = 'all_IHC_score_model_'+ model_labels[0]
        column_labels = ['img\ngroup 3+','img\ngroup 2+','img\ngroup 1+','img\ngroup 0']
        save_path = os.path.join(args.train_path,"qulitative eval")

    # Set the size of each subplot
    subplot_size = 3  # Adjust this value to control the size of each subplot
    fig_width = subplot_size * num_cols+ 1.0 
    fig_height = subplot_size * num_rows 
    # Create a figure with a size that accommodates the subplots and labels
    fig, axes = plt.subplots( num_rows, num_cols, figsize=(fig_width, fig_height))

    for ax in axes.ravel():
        ax.set_aspect('equal')
            

    # Create subplots and labels
        for j in range(num_cols):
            for i in range(num_rows):
                index = j * num_rows + i
                if index < len(img_arr):
                    axes[i, j].imshow(img_arr[index])
                    axes[i, j].get_xaxis().set_visible(False)
                    axes[i, j].get_yaxis().set_visible(False)
                        

        # Labels for columns at the very top
        for j, label in enumerate(column_labels):
            ax = axes[0, j]
            ax.set_title(label, fontsize=18, pad=10)  

        for i, label in enumerate(row_labels):
            ax = axes[i, 0]
            plt.gcf().text(0.065, 0.2+(i*0.25), label, fontsize=18)


        plt.subplots_adjust(wspace=0, hspace=0)
        plot_name =plot_name+ '_pred_examples.png'
        plt.savefig(os.path.join(self.save_path,plot_name), bbox_inches='tight')





































# get all images for publisching plot 
def get_publish_plot_img(args,images):

    plot_names =images["img_name"]

    plot_img_IHC = []
    plot_img_HE = []

    test_data = new_loader.stain_transfer_dataset( img_patch=0, set='test',args = args) 
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 
    transform_resize = T.Resize((256,256))


    for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader):
        
        if img_name[0] in plot_names:
            real_HE = transform_resize(real_HE)
            real_IHC = transform_resize(real_IHC)

            plot_img_IHC.append(real_IHC)
            plot_img_HE.append(real_HE)


    return plot_img_IHC, plot_img_HE

# predict all images with models for publishing plot
def predict_all_img(args,images,model_dict,gan_flag):
    plot_img_IHC, plot_img_HE = get_publish_plot_img(args=args, images=images)
    model_num =3
    img_arr = []
    num_samples = len(plot_img_IHC)
    print(len(plot_img_IHC))

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
        
        num_models=0
        #cycle through models 
        for architecture_name in model_dict:
  
            for version in model_dict[architecture_name]:

                args.model = architecture_name
                args.type = version
                
                model, model_name = utils.load_model(args)
                model = model.to(args.device)

                model_dir = os.path.join(Path.cwd(),"masterthesis_results")
                if gan_flag == True:
                    model_dir = os.path.join(model_dir,"Pix2Pix")

                train_path = os.path.join(model_dir,model_name)
                if os.path.isdir(train_path):
                    print(architecture_name)
                    best_model_weights = os.path.join(train_path,'final_weights_gen.pth')
                    model.load_state_dict(torch.load(best_model_weights))
                    model = model.to(args.device)
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
                num_models = num_models+1
                #fake_IHC = torch.from_numpy(fake_IHC)
                img_arr.append(fake_IHC)
                print('append predict')
                #num_models = num_models+1

    return img_arr, num_samples, num_models

def get_publishing_plot(args,images,model_dict,gan_flag,save_path,plot_name):
    img_arr, num_samples, num_models = predict_all_img(args, images, model_dict,gan_flag)
    images = img_arr  

    num_rows = num_samples  
    num_cols = num_models+2

    column_labels = ['HE \n Input','IHC \n Target']
    row_labels = ['img\ngroup 0','img\ngroup 1+','img\ngroup 2+','img\ngroup 3+']
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
        ax.set_title(label, fontsize=18, pad=10)  

    for i, label in enumerate(row_labels):
        ax = axes[i, 0]
        plt.gcf().text(0.04, 0.195+(i*0.195), label, fontsize=18)

 

    plt.subplots_adjust(wspace=0, hspace=0.01)
    plt.savefig(os.path.join(save_path,plot_name))
    return fig