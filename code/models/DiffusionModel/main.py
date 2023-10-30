import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)
import utils
import eval
import torch
import Framework_Diffusion
import time
from pathlib import Path
from Unet_diff import UNet
import loader
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import random
import numpy as np
from Diffusion_model import Diffusion
import torch.nn as nn
# --------------------------- load Parameters from config ----------------------------------
config_path = os.path.join(Path.cwd(),'code\\models\\DiffusionModel\\config.yaml')
params = utils.get_config_from_yaml(config_path)

train = True
test = True
training_time = 0

if train == True:
    if params['gen_architecture']== "base_Unet":
         model = UNet()
    # load model 
    if params['trained_model_path']!= "None":
        model.load_state_dict(torch.load(params['trained_model_path']))

    # --------------------------- intitialise cycle_Gan ----------------------------------------
    model = Framework_Diffusion.model(params=params, net = model)
    # --------------------------- Train Network ------------------------------------------------
    start = time.time()
    gen = model.fit()
    stop = time.time()


    # ------------------------------------------------------------------------------------------
    # save the trained model 
    # ------------------------------------------------------------------------------------------
    training_time = (stop-start)/60
    output_folder_path = os.path.join(params['output_path'],params['output_folder'])
    model_path = os.path.join(output_folder_path,params['model_name'])
    config_path =  os.path.join(output_folder_path,'config.yaml')

    utils.save_config_in_dir(config_path, params)
    torch.save(gen.state_dict(), model_path)
if test == True:
# ------------------------------------------------------------------------------------------
# Testing 
# ------------------------------------------------------------------------------------------

    model = UNet().to(params['device'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    diffusion = Diffusion(noise_steps=params['noise_steps'],beta_start=params['beta_start'],beta_end=params['beta_end'],img_size=params['img_size'],device=params['device'])  

    output_folder_path = os.path.join(params['output_path'],params['output_folder'])
    model_path = os.path.join(output_folder_path,params['model_name'])
    config_path =  os.path.join(output_folder_path,'config.yaml')
    test_path = params['test_dir']
    HE_img_dir = os.path.join(test_path,'HE')
    IHC_img_dir = os.path.join(test_path,'IHC')
    result_dir = os.path.join(output_folder_path,'result.txt')
    params = params
    train_time = training_time
    result_dir = os.path.join(output_folder_path,'result.txt')
    os.mkdir(os.path.join(os.path.join(params['output_path'],params['output_folder']),"test_plots"))

    # set up result vector 
    result = {}
    result['epoch'] = []
    result['ssim_mean'] = []
    result['ssim_std'] = []
    result['psnr_mean'] = []
    result['psnr_std'] = []
    result['training_time_min'] = train_time
    result['mse_mean']=[]
    result['mse_std']=[]


    for epoch in range(params['num_test_epochs']):
                
                result['epoch'].append(epoch)
                test_data = loader.stain_transfer_dataset(  img_patch= epoch,
                                                            img_size= params['img_size'],
                                                            HE_img_dir = HE_img_dir,
                                                            IHC_img_dir = IHC_img_dir,
                                                            params=params
                                                            )
                
                test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

                # ------ set up ssim and psnr ----------------------------------------------------
                ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
                psnr = PeakSignalNoiseRatio().cuda()
                MSE_LOSS = nn.MSELoss().cuda()

                
                # ------ set up ssim and psnr ----------------------------------------------------
                ssim_list = []
                psnr_list = []
                mse_list= []


                # get random instances 
                randomlist = []
                for i in range(0,params['plots_per_epoch']):
                    n = random.randint(1,len(test_data_loader))
                    randomlist.append(n)

                for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader):
                    fake_IHC = diffusion.sample(model , n=real_IHC.shape[0],y=real_HE)
                    ssim_score = float(ssim(fake_IHC, real_IHC))
                    psnr_score = float(psnr(fake_IHC, real_IHC))
                    mse_score = float(MSE_LOSS(fake_IHC, real_IHC))

                    if i in randomlist:
                    
                        utils.plot_img_set( real_HE=real_HE,
                                            fake_IHC=fake_IHC,
                                            real_IHC=real_IHC,
                                            i=i,
                                            params = params,
                                            img_name = img_name,
                                            step = 'test',
                                            epoch = epoch )
                
                    
                    ssim_list.append(ssim_score)
                    psnr_list.append(psnr_score)
                    mse_list.append(mse_score)
                    
                    del real_HE
                    del fake_IHC
                    del real_IHC

                result['mse_mean'].append(np.mean(mse_list))
                result['mse_std'].append(np.std(mse_list))

                result['ssim_mean'].append(np.mean(ssim_list))
                result['ssim_std'].append(np.std(ssim_list))

                result['psnr_mean'].append(np.mean(psnr_list))
                result['psnr_std'].append(np.std(psnr_list))


            
    result['total_MSE_mean'] = np.mean( result['mse_mean'])
    result['total_SSIM_mean'] = np.mean( result['ssim_mean'])
    result['total_PSNR_mean'] = np.mean( result['psnr_mean'])

    # write file
    with open(result_dir, 'w') as f: 
        for key, value in result.items(): 
            f.write('%s:%s\n' % (key, value))

    # close file
    f.close()
