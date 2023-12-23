import utils
import argparse
import os
from pathlib import Path
import numpy as np
import new_loader
import torch 
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import torch.nn as nn
import pickle

def my_args():
    parser = argparse.ArgumentParser()
    #
    # models and types:
    # - U_Net ==> type : S /S+att /M /M+att /L /L+att
    # - ViT ==> type : S  /M
    # - Swin ==> type : S
    # - Diffusion
    # - Classifier
    #
    # gan_framework:
    # - pix2pix
    # - score_gan
    
    parser.add_argument('--model', type=str, default="", help='model architecture')
    parser.add_argument('--type', type=str, default="", help='scope of the model S or M or L')
    parser.add_argument('--diff_noise_steps', type=int, default=1000, help='Image size')
    parser.add_argument('--gan_framework', type=str, default="None", help='set a gan framework')

    # Optimizer
    parser.add_argument('--lr', type=float, default=3e-5, help='learining rate')
    parser.add_argument('--beta1', type=float, default=0.5 , help='beta1 for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam optimizer')

    # training 
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--img_resize', action='store_true', default=False, help='resize image to 256')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--img_transforms', type=list, default=["colorjitter",'horizontal_flip','vertical_flip'], help='choose image transforms from normalize,colorjitter,horizontal_flip,grayscale')
    parser.add_argument('--num_epochs', type=int, default=100, help='epoch num')
    parser.add_argument('--decay_epoch', type=int, default=80, help='decay epoch num')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--device', type=str, default="cuda", help='device')


    # Loss
    parser.add_argument('--gaus_loss', action='store_true', default=False, help='activate gausian blurr loss')
    parser.add_argument('--ssim_loss', action='store_true', default=False, help='activate ssim  loss')
    parser.add_argument('--hist_loss', action='store_true', default=False, help='activate histogram loss')

    # Data dirs
    parser.add_argument('--train_data', type=str, default='C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/Data_set_BCI_challange/train', help='directory to the train data')
    parser.add_argument('--test_data', type=str, default='C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/Data_set_BCI_challange/val', help='directory to the test data')
    
    parser.add_argument('--train_path', type=str, default='', help='directory to the training')
    parser.add_argument('--train_eval_path', type=str, default='', help='directory to the train eval file')
    parser.add_argument('--test_eval_path', type=str, default='', help='directory to the test eval file')
    parser.add_argument('--tp_path', type=str, default='', help='directory to the train plots')
    parser.add_argument('--c_path', type=str, default='', help='directory to the checkpoints of the training')


    # Testing 
    parser.add_argument('--quant_eval_only', action='store_true', default=False, help='flag for only test')
    parser.add_argument('--qual_eval_only', action='store_true', default=False, help='flag for only classifer')


    return parser.parse_args() 


args = my_args()

args.model ="U_Net"
args.type ="L"
args.gaus = True
args.gan_framework = "None"
args.img_resize = True

model ,model_framework, model_arch, model_specs = utils.build_model(args)

args, model_name, train_plot_eval, test_plot_eval = utils.set_paths(args, model_framework, model_arch, model_specs)
result_dir = os.path.join(Path.cwd(),"masterthesis_results")
train_path = os.path.join(result_dir,model_name)
checkpoint_path =os.path.join(train_path,"checkpoints")

test_plot_eval = {}
test_plot_eval['MSE']= []
test_plot_eval['SSIM']= []
test_plot_eval['PSNR']= []
test_plot_eval['x']= []

def get_test_scores(args, model):
        
    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    PSNR = PeakSignalNoiseRatio().to(args.device)
    MSE = nn.MSELoss().to(args.device)
    model.to(args.device)
    model.eval()

    
    data_set = new_loader.stain_transfer_dataset( img_patch=0, set='test', args=args) 
    loader = DataLoader(data_set, batch_size=1, shuffle=False) 

    mse_list = []
    ssim_list = [] 
    psnr_list = []
    
    for i, (real_HE, real_IHC, img_name) in enumerate(loader):
        
        with torch.no_grad():
            fake_IHC = model(real_HE)

        mse_score = MSE(real_IHC,fake_IHC)
        ssim_score =  SSIM(real_IHC,fake_IHC)
        psnr_score = PSNR(real_IHC,fake_IHC)

        mse_list.append(mse_score.item())
        ssim_list.append(ssim_score.item())
        psnr_list.append(psnr_score.item())

        #mean for each epoch 
        mse_mean = np.mean(mse_list)
        ssim_mean = np.mean(ssim_list)
        psnr_mean = np.mean(psnr_list)

    return mse_mean, ssim_mean, psnr_mean





for x in range(0,96,5):
    weights_name = "gen_G_weights_"+str(x)+".pth"
    weights_path = os.path.join(checkpoint_path,weights_name)
    model.load_state_dict(torch.load(weights_path))
    print(' ---------------------------------------------- ')
    print('pretrained ' +model_name+'  weights loaded')

    mse_mean, ssim_mean, psnr_mean = get_test_scores(args, model)
    test_plot_eval['MSE'].append(mse_mean)
    test_plot_eval['SSIM'].append(ssim_mean)
    test_plot_eval['PSNR'].append(psnr_mean)
    test_plot_eval['x'].append(x)

with open(os.path.join(train_path,'test_plot_eval_resize'), "wb") as fp:   
    pickle.dump(test_plot_eval, fp)


