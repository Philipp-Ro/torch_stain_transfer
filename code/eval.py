import os
import torch
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import new_loader
import numpy as np
import random

import torch.nn as nn
from pathlib import Path
import pickle
import plot_utils


class test_network():
    def __init__(self, args, model, model_name):
        self.args = args

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
        self.psnr = PeakSignalNoiseRatio().to(args.device)
        self.MSE_LOSS = nn.MSELoss().to(args.device)

        self.model = model.to(args.device)

        self.results_dir = os.path.join(Path.cwd(),"masterthesis_results")
        model_dir = os.path.join(self.results_dir, model_name)

        self.tr_path = os.path.join(model_dir,'train_result')
        
        with open(self.tr_path, "rb") as fp:   # Unpickling
            self.train_eval = pickle.load(fp)

        self.count =  len(self.train_eval['train_mse'])
        testing_name = "Testing_"+str(self.count)+"_epochs"
        testing_path = os.path.join(model_dir,testing_name)
        os.mkdir(testing_path)

        self.model_path = os.path.join(model_dir,'final_weights_gen.pth')
        self.resultfile_path = os.path.join(testing_path,'result.txt')
        self.testplots_folder = os.path.join(testing_path,"plots")
        os.mkdir(self.testplots_folder)

    def eval(self):
        self.model.load_state_dict(torch.load(self.model_path))
        

        # set up result vector 
        result = {}
        result['epoch'] = []
        result['ssim_mean'] = []
        result['ssim_std'] = []
        result['psnr_mean'] = []
        result['psnr_std'] = []
        result['mse_mean'] = []
        result['mse_std']= []



        for epoch in range(self.args.num_test_epochs):
    
            result['epoch'].append(epoch)
            test_loader = new_loader.stain_transfer_dataset( img_patch=epoch, set='test',args = self.args) 
            
            test_data_loader = DataLoader(test_loader, batch_size=1, shuffle=False) 

            # ------ set up ssim and psnr ----------------------------------------------------
            ssim_list = []
            psnr_list = []
            mse_list= []

            # ----------- get plots 
            plot_list = []
            if self.args.testplot_idx != []:
                plot_list = self.args.testplot_idx
            else:
                # get random instances 
                for i in range(0,2):
                    n = random.randint(1,len(test_data_loader))
                    plot_list.append(n)

            for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader):

                fake_IHC = self.model(real_HE)
                ssim_score = float(self.ssim(fake_IHC, real_IHC))
                psnr_score = float(self.psnr(fake_IHC, real_IHC))
                mse_score = float(self.MSE_LOSS(fake_IHC, real_IHC))

                if i in plot_list and epoch ==7:
                    plot_utils.plot_img_set( real_HE = real_HE,
                                    fake_IHC=fake_IHC,
                                    real_IHC=real_IHC,
                                    save_path = self.testplots_folder,
                                    img_name = img_name,
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
        with open(self.resultfile_path, 'w') as f: 
            for key, value in result.items(): 
                f.write('%s:%s\n' % (key, value))

        # close file
        f.close()





