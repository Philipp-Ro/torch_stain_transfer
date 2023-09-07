import os
import torch
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import loader
import numpy as np
import random
import utils


class test_network():
    def __init__(self,model,params,train_time):
        self.output_folder_path = os.path.join(params['output_path'],params['output_folder'])
        self.model_path = os.path.join(self.output_folder_path,params['model_name'])
        self.config_path =  os.path.join(self.output_folder_path,'config.yaml')
        test_path = params['test_dir']
        self.HE_img_dir = os.path.join(test_path,'HE')
        self.IHC_img_dir = os.path.join(test_path,'IHC')
        self.result_dir = os.path.join(self.output_folder_path,'result.txt')
        self.params = params
        self.train_time = train_time
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(params['device'])
        self.psnr = PeakSignalNoiseRatio().to(params['device'])

        self.model = model


    def fit(self):
        self.model.load_state_dict(torch.load(self.model_path))
        #self.model.eval()    

        # set up result vector 
        result = {}
        result['epoch'] = []
        result['ssim_mean'] = []
        result['ssim_std'] = []
        result['psnr_mean'] = []
        result['psnr_std'] = []
        result['training_time_min'] = self.train_time

        for epoch in range(self.params['num_test_epochs']):
    
            result['epoch'].append(epoch)
            test_data = loader.stain_transfer_dataset(  img_patch= epoch,
                                                        params= self.params,
                                                        HE_img_dir = self.HE_img_dir,
                                                        IHC_img_dir = self.IHC_img_dir,
                                                        )
            
            test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

            # ------ set up ssim and psnr ----------------------------------------------------
            ssim_list = []
            psnr_list = []

            # ----------- get plots 
            plot_list = []
            if self.params['test_idx'] != []:
                plot_list = self.params['test_idx']
            else:
                # get random instances 
                for i in range(0,self.params['plots_per_epoch']):
                    n = random.randint(1,len(test_data_loader))
                    plot_list.append(n)

            for i, (real_HE, real_IHC, img_name) in enumerate(test_data_loader):

                fake_IHC = self.model(real_HE)
                print(i)
                ssim_score = float(self.ssim(fake_IHC, real_IHC))
                psnr_score = float(self.psnr(fake_IHC, real_IHC))
                if i in plot_list:
                  
                    utils.plot_img_set( real_HE=real_HE,
                                        fake_IHC=fake_IHC,
                                        real_IHC=real_IHC,
                                        i=i,
                                        params = self.params,
                                        img_name = img_name,
                                        step = 'test',
                                        epoch = epoch )
            

                ssim_list.append(ssim_score)
                psnr_list.append(psnr_score)
                
                del real_HE
                del fake_IHC
                del real_IHC

            result['ssim_mean'].append(np.mean(ssim_scores))
            result['ssim_std'].append(np.std(ssim_scores))

            result['psnr_mean'].append(np.mean(psnr_scores))
            result['psnr_std'].append(np.std(psnr_scores))

        # open file for writing
        f = open(self.result_dir,"w")

        # write file
        f.write( str(result) )

        # close file
        f.close()





