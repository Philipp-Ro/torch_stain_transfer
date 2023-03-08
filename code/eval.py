import os
import torch
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import loader
import numpy as np
import random
import utils
import conv_models
import transformer_models

class test_network():
    def __init__(self,params):
        self.output_folder_path = os.path.join(params['output_path'],params['output_folder'])
        self.model_path = os.path.join(self.output_folder_path,params['model_name'])
        self.config_path =  os.path.join(self.output_folder_path,'/config.yaml')
        test_path = params['test_dir']
        self.HE_img_dir = os.path.join(test_path,'HE')
        self.IHC_img_dir = os.path.join(test_path,'IHC')
        self.result_dir = os.path.join(self.output_folder_path,'result.txt')
        self.params = params

        self.model = initialize_gen_model(params)


    def fit(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()    

        # set up result vector 
        result = {}
        result['epoch'] = []
        result['ssim_mean'] = []
        result['ssim_std'] = []
        result['psnr_mean'] = []
        result['psnr_std'] = []

        for epoch in range(self.params['num_epochs']):
    
            result['epoch'].append(epoch)
            test_data = loader.stain_transfer_dataset(  img_patch= epoch,
                                                        norm = self.params['norm'],
                                                        grayscale = self.params['grayscale'],
                                                        HE_img_dir = self.HE_img_dir,
                                                        IHC_img_dir = self.IHC_img_dir,
                                                        img_size= self.params['img_size'],
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
            for i in range(0,self.params['plots_per_epoch']):
                n = random.randint(1,len(test_data_loader))
                randomlist.append(n)

            for i, (real_HE,real_HE_norm, real_IHC,real_IHC_norm, img_name) in enumerate(test_data_loader):
                fake_IHC = self.model(real_HE)
                fake_IHC = fake_IHC+1
                fake_IHC = fake_IHC*0.5

                if i in randomlist:
                    print(img_name)
                    #print(img_name.type())
                    utils.plot_img_set(real_HE, real_IHC, fake_IHC, i,self.params,img_name)
            
                
                ssim_scores.append(ssim(fake_IHC, real_IHC).item())
                psnr_scores.append(psnr(fake_IHC, real_IHC).item())
                

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


def initialize_gen_model(params):
    if params['gen_architecture'] == 'conv':
        gen_test = conv_models.GeneratorResNet( in_channels= params['in_channels'],
                                                num_residual_blocks = params['num_resnet']
                                            )
        
    if params['gen_architecture'] == 'trans':
        gen_test = transformer_models.Generator(    img_size= params['img_size'][0],
                                                    embedding_dim=0,
                                                    patch_size=params['patch_size'],
                                                    in_channels=params['in_channels'],
                                                    dropout_embedding=params['dropout_embedding'],
                                                    nhead= params['nhead'],
                                                    num_layers=params['num_layers']
                                            )
    gen_test = gen_test.cuda()

    return gen_test


