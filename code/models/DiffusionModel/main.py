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
# --------------------------- load Parameters from config ----------------------------------
config_path = os.path.join(Path.cwd(),'code\\models\\DiffusionModel\\config.yaml')
params = utils.get_config_from_yaml(config_path)

# --------------------------- intitialise cycle_Gan ----------------------------------------
model = Framework_Diffusion.model(params=params)
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

# set up result vector 
result = {}
result['epoch'] = []
result['ssim_mean'] = []
result['ssim_std'] = []
result['psnr_mean'] = []
result['psnr_std'] = []
result['training_time_min'] = train_time

for epoch in range(params['num_test_epochs']):
            
            result['epoch'].append(epoch)
            test_data = loader.stain_transfer_dataset(  img_patch= epoch,
                                                        params= params,
                                                        HE_img_dir = HE_img_dir,
                                                        IHC_img_dir = IHC_img_dir,
                                                        )
            
            test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

            # ------ set up ssim and psnr ----------------------------------------------------
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
            ssim = ssim.cuda()
            ssim_scores = []

            psnr = PeakSignalNoiseRatio()
            psnr = psnr.cuda()
            psnr_scores = []

            # get random instances 
            randomlist = []
            for i in range(0,params['plots_per_epoch']):
                n = random.randint(1,len(test_data_loader))
                randomlist.append(n)

            for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader):
                fake_IHC = diffusion.sample(model , n=real_IHC.shape[0],y=real_IHC)
                if "normalise" in params["preprocess_IHC"]:
                    fake_IHC = utils.denomalise(params['mean_IHC'], params['std_IHC'],fake_IHC)
                    real_IHC = utils.denomalise(params['mean_IHC'], params['std_IHC'],real_IHC)

                if "normalise" in params["preprocess_HE"]:
                    real_HE = utils.denomalise(params['mean_HE'], params['std_HE'],real_HE)

                if i in randomlist:
                  
                    utils.plot_img_set( real_HE=real_HE,
                                        fake_IHC=fake_IHC,
                                        real_IHC=real_IHC,
                                        i=i,
                                        params = params,
                                        img_name = img_name,
                                        step = 'test',
                                        epoch = epoch )
            
                
                ssim_scores.append(ssim(fake_IHC, real_IHC).item())
                psnr_scores.append(psnr(fake_IHC, real_IHC).item())
                

            result['ssim_mean'].append(np.mean(ssim_scores))
            result['ssim_std'].append(np.std(ssim_scores))

            result['psnr_mean'].append(np.mean(psnr_scores))
            result['psnr_std'].append(np.std(psnr_scores))

                

# open file for writing
f = open(result_dir,"w")

# write file
f.write( str(result) )

# close file
f.close()