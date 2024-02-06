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
    if 'Diffusion' not in args.model:
        axs[0].plot(test_eval['x'], test_eval['MSE'],label='test MSE')
    axs[0].legend(loc="upper right",fontsize='xx-small')
    axs[0].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
    axs[0].set_title('MSE',loc='left')

    axs[1].plot(train_eval['x'], train_eval['SSIM'],label='train SSIM')
    if 'Diffusion' not in args.model:
        axs[1].plot(test_eval['x'], test_eval['SSIM'],label='test SSIM')
    axs[1].legend(loc="lower right",fontsize='xx-small')
    axs[1].set_xlabel(xlabel='epoch',loc='right',labelpad=2)
    axs[1].set_title('SSIM',loc='left')

    axs[2].plot(train_eval['x'], train_eval['PSNR'],label='train PSNR')
    if'Diffusion' not in args.model:
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
        diffusion = Diffusion(noise_steps=args.diff_noise_steps,img_size=256,device=args.device)

    test_data = new_loader.stain_transfer_dataset( img_patch=0, set='test',args = args) 
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

    for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader) :
                 
        if img_name[0] in img_names :
            real_HE = transform_resize(real_HE)
            real_IHC = transform_resize(real_IHC)

            if args.model == 'Diffusion':

                fake_IHC = diffusion.sample(model , n=real_IHC.shape[0], y=real_HE)
                fake_IHC = fake_IHC.cpu().detach().numpy()
                fake_IHC = np.squeeze(fake_IHC)
                fake_IHC = np.transpose(fake_IHC, (1, 2, 0))
                img_arr.append(fake_IHC)


            elif model == 'source':
                fake_IHC = []

                real_HE_plot = real_HE.cpu().detach().numpy()
                real_HE_plot = np.squeeze(real_HE_plot )
                real_HE_plot = np.transpose(real_HE_plot, (1, 2, 0))
                img_arr.append(real_HE_plot)

            elif model == 'target':
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
    
def get_imgs_for_all_models(args, model_list, img_names):
    

    model_labels = []

    # get HE imgs
    model = 'source'
    img_arr_source = get_imgs(args, img_names, model)
    img_arr =  img_arr_source
    model_labels.append('HE\nInput')
    

    # get IHC imgs
    model = 'target'
    img_arr_target = get_imgs(args, img_names, model)
    img_arr = np.vstack((img_arr, img_arr_target))
    model_labels.append('IHC\nTarget')

    result_dir = os.path.join(Path.cwd(),"masterthesis_results")
    # get images from all networks in model_list
    for model_name in model_list:
        model_dir = os.path.join(result_dir,model_name)
  
        if 'U_Net' in model_name:
            args.model = 'U_Net'
            if '3step' in model_name:
                args.type = 'S'
            if '4step' in model_name:
                args.type = 'M'       
            if '5step' in model_name:
                args.type = 'L'  

        if 'ViT' in model_name:
            args.model = 'ViT'
            if '1_block' in model_name:
                args.type = 'S'
            if '2_block' in model_name:
                args.type = 'M'

        if 'Swin_T' in model_name:
            args.model = 'Swin'
            args.type = 'S'

        if 'diffusion' in model_name:
            args.model = 'Diffusion'
            args.type = 'M'

        model_label_name = args.model +'\n'+args.type

        if 'Pix2Pix' in model_name:
            model_label_name = 'pix2pix\n'+model_label_name
            args.gan_framework = 'pix2pix'

        model_labels.append(model_label_name)


        model ,model_framework, model_arch, model_specs = utils.build_model(args)
        args.train_path = model_dir
        trained_model = utils.load_model_weights(args, model, model_name)
        trained_model = model.to(args.device)
        #trained_model.eval()

        img_arr_model = get_imgs(args, img_names, trained_model)
        #img_arr = np.concatenate((img_arr, img_arr_model), axis=1)
        img_arr = np.vstack((img_arr,  img_arr_model))

    return img_arr, model_labels

def save_plot_for_models(args, model_list, IHC_score):

    if IHC_score == '0':
        img_names = ['00292_train_0.png','00549_train_0.png','01434_train_0.png','01810_train_0.png']
        plot_name = 'all_models_IHC_score_0'
        column_labels = ['img 1\nIHC '+IHC_score,'img 2\nIHC '+IHC_score,'img 3\nIHC '+IHC_score,'img 4\nIHC '+IHC_score]

    if IHC_score == '1+':
        img_names = ['02042_train_1+.png', '01995_train_1+.png','02867_train_1+.png','03759_train_1+.png']
        plot_name = 'all_models_IHC_score_1+'
        column_labels = ['img 1\nIHC '+IHC_score,'img 2\nIHC '+IHC_score,'img 3\nIHC '+IHC_score,'img 4\nIHC '+IHC_score]

    if IHC_score == '2+':
        img_names = ['01879_train_2+.png', '02555_train_2+.png','03078_train_2+.png','03877_train_2+.png']
        plot_name = 'all_models_IHC_score_2+'
        column_labels = ['img 1\nIHC '+IHC_score,'img 2\nIHC '+IHC_score,'img 3\nIHC '+IHC_score,'img 4\nIHC '+IHC_score]

    if IHC_score == '3+':
        img_names = ['02995_train_3+.png', '02171_train_3+.png', '01671_train_3+.png', '00788_train_3+.png']
        plot_name = 'all_models_IHC_score_3+'
        column_labels = ['img 1\nIHC '+IHC_score,'img 2\nIHC '+IHC_score,'img 3\nIHC '+IHC_score,'img 4\nIHC '+IHC_score]


    if IHC_score == 'all':
        img_names = ['00292_train_0.png','00323_train_1+.png','00605_train_2+.png','01190_train_3+.png']
        plot_name = 'all_IHC_score_model_'+ args.model+'_'+args.type
        column_labels = ['img\nscore 0','img\nscore 1+','img\nscore 2+','img\nscore 3+']
        
    if len(model_list)==1:
        save_path = os.path.join(Path.cwd(),"qualitative eval")
    else:
        save_path = os.path.join(Path.cwd(),'all_net_plots')

    img_arr, model_labels = get_imgs_for_all_models(args, model_list, img_names)
    
    num_rows = len(model_labels) 
    num_cols = len(img_names)


    #row_labels = model_labels
    # Set the size of each subplot
    subplot_size = 3  # Adjust this value to control the size of each subplot
    fig_width = subplot_size * num_cols+ 1.0 
    fig_height = subplot_size * num_rows 
    # Create a figure with a size that accommodates the subplots and labels
    fig, axes = plt.subplots( num_rows, num_cols, figsize=(fig_width, fig_height))

    for ax in axes.ravel():
        ax.set_aspect('equal')
            
    # Create subplots and labels
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < len(img_arr):
                axes[i, j].imshow(img_arr[index])
                if i == 0:
                    axes[i, j].set_title(column_labels[j])
                if j == 0:
                    axes[i, j].set_ylabel(model_labels[i], rotation=0, size='large')
                    axes[i, j].yaxis.set_label_coords(-.2, .5)

                axes[i, j].xaxis.set_tick_params(labelbottom=False)
                axes[i, j].yaxis.set_tick_params(labelleft=False)
  
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plot_name ='masterthesis_results/'+plot_name+ 'pred_examples.png'
    plt.savefig(os.path.join(Path.cwd(),plot_name), bbox_inches='tight')



def get_boxplot_for_scores(args, metric_name, train_list, test_list, path):
    train_plot_data = [ train_list['group_0'],train_list['group_1'], train_list['group_2'], train_list['group_3']]
    test_plot_data = [ test_list['group_0'], test_list['group_1'], test_list['group_2'], test_list['group_3']]

    score_labels = [ '0', '1+', '2+', '3+']
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)

    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
        plt.plot([], c=color_code, label=label)
        plt.legend()


    bp_train = ax.boxplot(train_plot_data , positions=np.array(np.arange(len(train_plot_data)))*2.0-0.35, widths=0.6)
    bp_test = ax.boxplot(test_plot_data, positions=np.array(np.arange(len(test_plot_data)))*2.0+0.35, widths=0.6)


    
    # setting colors for each groups
    define_box_properties(bp_train , '#1f77b4', 'train')
    define_box_properties(bp_test, '#ff7f0e', 'test')
    ax.get_xaxis().tick_bottom()
    plt.xticks(np.arange(0, len(score_labels) * 2, 2), score_labels)
    font = {'family':'serif','color':'black','size':20}

    plt.ylabel(metric_name, fontdict = font)
    plt.xlabel('IHC score', fontdict = font)
    plot_name =args.model +'_'+args.type +'_'+metric_name+'_plot.png'
    plt.savefig(os.path.join(path, plot_name) ,dpi=300)
































