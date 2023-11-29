import os
import torch
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import new_loader
import numpy as np
import time
import utils
import torch.nn as nn
from pathlib import Path
import pickle
import plot_utils
from architectures.Diffusion_model import Diffusion


class test_network():
    def __init__(self, args, model, model_name):
        self.args = args
        self.model_name = model_name

        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
        self.PSNR = PeakSignalNoiseRatio().to(args.device)
        self.MSE = nn.MSELoss().to(args.device)

        self.model = model.to(args.device)

        self.results_dir = os.path.join(Path.cwd(),"masterthesis_results")
        model_dir = os.path.join(self.results_dir, model_name)
        checkpoint_dir = os.path.join(model_dir,"checkpoints")

        train_eval_plot_path = os.path.join(model_dir,'train_plot_eval')
        test_eval_plot_path = os.path.join(model_dir,'test_plot_eval')
        if "diff" not in model_name:
            # load train data
            with open(train_eval_plot_path, "rb") as fp:   
                    self.train_plot_eval = pickle.load(fp)

            with open(test_eval_plot_path, "rb") as fp:   
                    self.test_plot_eval = pickle.load(fp)
            
            

            num_epochs = len(self.train_plot_eval['MSE'])

            testing_name = 'Testing_'+str(num_epochs)+'_epochs'
            self.testing_path = os.path.join(model_dir,testing_name)
            os.mkdir(self.testing_path)

            self.train_eval_path = os.path.join(self.testing_path,'train_result.txt')
            self.test_eval_path = os.path.join(self.testing_path,'test_result.txt')
            self.val_eval_path = os.path.join(self.testing_path,'val_result.txt')

            plot_utils.plot_trainresult(self.args, self.testing_path, self.train_plot_eval, self.test_plot_eval)
            error_score = 2
            for n in range(len(self.test_plot_eval['x'])):
                current_error_score = self.test_plot_eval['MSE'][n]+(1-self.test_plot_eval['SSIM'][n])
                if current_error_score < error_score:
                    error_score = current_error_score
                    model_num = self.test_plot_eval['x'][n]
        

            model_name_weights = "gen_G_weights_"+ str(model_num) +".pth"
            print('model epoch '+ str(model_num) +' choosen as best weights')
            weight_path =os.path.join(checkpoint_dir,model_name_weights)
            self.model.load_state_dict(torch.load(weight_path))
            # train metric plots
            model_name_epoch =  'model_final_weights_'+str(num_epochs)+'_epochs.pth'
            final_model_path = os.path.join(self.testing_path,model_name_epoch)
            torch.save(model.state_dict(), final_model_path)
        else:
            testing_name = 'Testing_100_epochs'
            self.testing_path = os.path.join(model_dir,testing_name)
            os.mkdir(self.testing_path)
            self.diffusion = Diffusion(noise_steps=args.diff_noise_steps,img_size=args.img_size,device=args.device) 
  

    def get_full_eval(self, data_set, model, group_wise, train_time ):
        result_total = utils.init_eval()
        prediction_time = []
        result_total['train_time'] = train_time
        num_patches = ((1024 * 1024) // self.args.img_size**2)-1
        if data_set =="train":
            start = 0
            stop = num_patches 
            save_path = os.path.join(self.testing_path,'train_result.txt')
            print('---------------------------------------------- ')
            print('TEST ON TRAIN')
            print('---------------------------------------------- ')
       
        if data_set =="test":  
            start = 0
            stop = num_patches
            save_path = os.path.join(self.testing_path,'test_result.txt')
            print('---------------------------------------------- ')
            print('TEST ON TEST')
            print('---------------------------------------------- ')

                
        for epoch in range(start,stop,1):
            data_set_init = new_loader.stain_transfer_dataset( img_patch=epoch, set=data_set, args=self.args) 
            loader = DataLoader(data_set_init, batch_size=1, shuffle=False) 

            mse_list, ssim_list, psnr_list = utils.init_epoch_eval_list()
            show_epoch  = 'Epoch: '+str(epoch)
            print(show_epoch)

            for i, (real_HE, real_IHC, img_name) in enumerate(loader):
                # predict:
                
                if "diff" not in self.model_name:
                    start_pred = time.time()
                    fake_IHC = model(real_HE)
                    end_pred  = time.time()
                    prediction_time.append(end_pred - start_pred)
                else:
                    start_pred = time.time()
                    fake_IHC = self.diffusion.sample(model , n=real_IHC.shape[0], y=real_HE)
                    end_pred  = time.time()
                    prediction_time.append(end_pred - start_pred)

                # calculate metrics
                ssim_score = self.SSIM(fake_IHC, real_IHC)
                psnr_score = self.PSNR(fake_IHC, real_IHC)
                mse_score = self.MSE(fake_IHC, real_IHC)

                if group_wise:
                    mse_list = utils.append_score_to_group_list(mse_score.item() ,mse_list, img_name)
                    ssim_list = utils.append_score_to_group_list(ssim_score.item() ,ssim_list, img_name)
                    psnr_list = utils.append_score_to_group_list(psnr_score.item()  ,psnr_list, img_name)
                
                mse_list['total'].append(mse_score.item())
                ssim_list['total'].append(ssim_score.item())
                psnr_list['total'].append(psnr_score.item())

        for key_group,value in mse_list.items():
                result_total[key_group]['MSE_mean']=np.mean(mse_list[key_group])
                result_total[key_group]['SSIM_mean']=np.mean(ssim_list[key_group])
                result_total[key_group]['PSNR_mean']=np.mean(psnr_list[key_group])
                result_total[key_group]['num_img'] = len(mse_list[key_group])

        result_total['prediction_time'] = np.mean(prediction_time)
        utils.write_result_in_file(save_path, result_total, data_set)










