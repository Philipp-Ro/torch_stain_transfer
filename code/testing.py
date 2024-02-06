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
from torchvision.models import resnet50, ResNet50_Weights 
import torchvision.transforms as T
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from transformers import ViTForImageClassification

class test_network():
    def __init__(self, args, model):
        self.args = args
        
        # init metrics
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
        self.PSNR = PeakSignalNoiseRatio().to(args.device)
        self.MSE = nn.MSELoss().to(args.device)
        self.CE = torch.nn.CrossEntropyLoss()

        # init classifier
        
        #classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  
        #classifier.fc = torch.nn.Linear(classifier.fc.in_features, 4)
        #torch.nn.init.xavier_uniform_(classifier.fc.weight)
        model_name_or_path = 'google/vit-base-patch16-224-in21k'
        classifier = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=4)
        self.classifier = classifier.to(args.device)

        # init diffusion
        self.diffusion = Diffusion(noise_steps=args.diff_noise_steps,img_size=args.img_size,device=args.device)
        self.transform_resize = T.Resize((256,256))
        self.transform_resize_class=T.Resize((224,224))

        self.model = model.to(args.device)
        self.model.eval()



            
        

    def get_full_quant_eval(self, data_set, model, train_time ):

        # init quant eval path
        
        quant_eval_path = os.path.join(self.args.train_path,'quantitative eval 2')

        if not os.path.isdir(quant_eval_path):
            os.mkdir(quant_eval_path)

        #save model 
        model_name_epoch =  'best_model_test.pth'
        final_model_path = os.path.join(quant_eval_path, model_name_epoch)
        torch.save(model.state_dict(), final_model_path)

        # load train data
        if 'Diffusion' in self.args.model or self.args.img_resize == True:
            num_patches=1   
        else:
            num_patches = ((1024 * 1024) // self.args.img_size**2)

            with open(self.args.train_eval_path, "rb") as fp:   
                    self.train_plot_eval = pickle.load(fp)

            with open(self.args.test_eval_path, "rb") as fp:   
                    self.test_plot_eval = pickle.load(fp)

            # plot the train metrics
            plot_utils.plot_trainresult(self.args, quant_eval_path, self.train_plot_eval, self.test_plot_eval)

        # init eval 
        result_total = utils.init_eval()
        prediction_time = []
        result_total['train_time'] = train_time

        if data_set =="train":
            start = 0
            stop = num_patches 
            
            print('---------------------------------------------- ')
            print('QUANTITATIV EVALUATION ON TRAIN')
            print('---------------------------------------------- ')
       
        if data_set =="test":  
            start = 0
            stop = num_patches
            
            print('---------------------------------------------- ')
            print('QUANTITATIV EVALUATION ON TEST')
            print('---------------------------------------------- ')
 
        for epoch in range(start,stop,1):
            data_set_init = new_loader.stain_transfer_dataset( img_patch=epoch, set=data_set, args=self.args) 
            loader = DataLoader(data_set_init, batch_size=1, shuffle=False) 

            mse_list, ssim_list, psnr_list = utils.init_epoch_eval_list()

            for i, (real_HE, real_IHC, img_name) in enumerate(loader):
                
                if self.args.img_resize or 'Diffusion' in self.args.model:
                    # resize full image
                    real_HE = self.transform_resize(real_HE)
                    real_IHC = self.transform_resize(real_IHC)

                # predict:
                if self.args.model == 'Diffusion':
                    start_pred = time.time()
                    with torch.no_grad():
                        fake_IHC = self.diffusion.sample(model , n=real_IHC.shape[0], y=real_HE)
                    end_pred  = time.time()
                    prediction_time.append(end_pred - start_pred)
                else:
                    start_pred = time.time()
                    with torch.no_grad():
                        fake_IHC = model(real_HE)
                    end_pred  = time.time()
                    prediction_time.append(end_pred - start_pred)

                # calculate metrics
                ssim_score = self.SSIM(fake_IHC, real_IHC)
                psnr_score = self.PSNR(fake_IHC, real_IHC)
                mse_score = self.MSE(fake_IHC, real_IHC)
   
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

        for key_group,value in mse_list.items():
            result_total[key_group]['MSE_mean']=np.mean(mse_list[key_group])
            result_total[key_group]['MSE_var']=np.var(mse_list[key_group])
            result_total[key_group]['MSE_min']=min(mse_list[key_group])
            result_total[key_group]['MSE_max']=max(mse_list[key_group])

            result_total[key_group]['SSIM_mean']=np.mean(ssim_list[key_group])
            result_total[key_group]['SSIM_var']=np.var(ssim_list[key_group])
            result_total[key_group]['SSIM_min']=min(ssim_list[key_group])
            result_total[key_group]['SSIM_max']=max(ssim_list[key_group])

            result_total[key_group]['PSNR_mean']=np.mean(psnr_list[key_group])
            result_total[key_group]['PSNR_var']=np.var(psnr_list[key_group])
            result_total[key_group]['PSNR_min']=min(psnr_list[key_group])
            result_total[key_group]['PSNR_max']=max(psnr_list[key_group])

            result_total[key_group]['num_img']=len(mse_list[key_group])

        result_total['prediction_time'] = np.mean(prediction_time)
        mse_name = 'mse_list_'+data_set
        ssim_name = 'ssim_list_'+data_set
        psnr_name = 'psnr_list_'+data_set

        with open(os.path.join(quant_eval_path,mse_name), "wb") as fp:   
            pickle.dump(mse_list, fp)

        with open(os.path.join(quant_eval_path,ssim_name), "wb") as fp:   
            pickle.dump(ssim_list, fp)

        with open(os.path.join(quant_eval_path,psnr_name), "wb") as fp:   
            pickle.dump(psnr_list, fp)

        return result_total, mse_list, ssim_list, psnr_list


    def get_full_qual_eval(self, classifier_name, model, model_list):
        
            
        # init qual eval dir
        qual_eval_path = os.path.join(self.args.train_path,"qualitative eval 2")
        os.mkdir(qual_eval_path)

        #save model in test file 
        model_name_epoch =  'model_final_weights_gen.pth'
        final_model_path = os.path.join(self.args.train_path, model_name_epoch)
        #model_name_epoch =  'model_final_weights_'+str(self.args.num_epochs)+'_epochs.pth'
        #final_model_path = os.path.join(qual_eval_path, model_name_epoch)
        torch.save(model.state_dict(), final_model_path)

        print('---------------------------------------------- ')
        print('QUALITATIV EVALUATION ON TEST')
        print('---------------------------------------------- ')


        # set save names
        model_png_name = self.args.model +'_'+self.args.type
        result_path = os.path.join(Path.cwd(),"masterthesis_results")
        best_model_weights = os.path.join(result_path,'classifier_weights'+ classifier_name +'.pth')
        self.classifier.load_state_dict(torch.load(best_model_weights))
        predictions = []
        score_list = []

        # set images on max size 
        self.args.img_size = 1024

        for epoch in range(1):
            test_data = new_loader.stain_transfer_dataset( img_patch=0, set='test',args = self.args) 
            test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 
            for i, (real_HE, real_IHC,img_name) in enumerate(test_data_loader) :

                # resize full image
                real_HE = self.transform_resize(real_HE)
                real_IHC = self.transform_resize(real_IHC)

                # get IHC score target
                score = utils.get_IHC_score(img_name)

                # get IHC score prediction 
                if self.args.model == 'Diffusion':
                    fake_IHC = self.diffusion.sample(model , n=real_IHC.shape[0], y=real_HE)
                    outputs = self.classifier(fake_IHC)

                elif self.args.model == "Classifier":
                    fake_IHC = []
                    outputs = self.classifier(real_IHC)
                        
                else:
                    fake_IHC = model(real_HE)
                    fake_IHC_in = self.transform_resize_class(fake_IHC)
                    outputs = self.classifier(fake_IHC_in)

                # one hot encode gt
                # predict score 
                outputs = outputs.logits
     
                outputs = torch.squeeze(outputs)
                #####
                value, idx = torch.max(outputs, 0)

                predictions.append(idx.item())
                score_list.append(score.item())


        
        cm_display = metrics.ConfusionMatrixDisplay.from_predictions(y_true=np.array(score_list), y_pred=np.array(predictions), display_labels = ["score:0", "score:1+", "score:2+", "score:3+"],cmap=plt.cm.Blues,colorbar=False)
            
        class_names = ["score:0", "score:1+", "score:2+", "score:3+"]
        report = metrics.classification_report(np.array(score_list), np.array(predictions), target_names=class_names, output_dict=True)
        df = pd.DataFrame(report).transpose()
    
        classification_report_name  = "classification_matrix.txt"
        save_path_report =os.path.join(qual_eval_path,classification_report_name)

        with open(save_path_report, 'a') as f:
            df_string = df.to_string()
            f.write(df_string)

        # close file
        f.close()
        fig = plt.figure()
        cm_display.plot(cmap=plt.cm.Blues)
            
        conf_mat_name = model_png_name+'_conf_mat.png'
        if self.args.gan_framework == 'pix2pix':
            conf_mat_name = 'Pix2Pix_'+conf_mat_name
        if self.args.gan_framework == 'score_gan':
            conf_mat_name = 'score_gan_'+conf_mat_name

        save_path_conf_mat =os.path.join(qual_eval_path,conf_mat_name)
        plt.savefig(save_path_conf_mat,dpi=300)

        # plot images
        model_labels = []
        img_names = ['00292_train_0.png','00323_train_1+.png','00605_train_2+.png','01190_train_3+.png']
        column_labels = ['img\nscore 0','img\nscore 1+','img\nscore 2+','img\nscore 3+']

        # get HE imgs
        img_arr_source = plot_utils.get_imgs(args=self.args, img_names=img_names, model= 'source')
        img_arr =  img_arr_source
        model_labels.append('HE\nInput')
        

        # get IHC imgs
        img_arr_target = plot_utils.get_imgs(args=self.args, img_names=img_names, model= 'source')
        img_arr = np.vstack((img_arr, img_arr_target))
        model_labels.append('IHC\nTarget')

        # get generated pictures
        img_arr_model = plot_utils.get_imgs(args=self.args, img_names=img_names, model= model)
        img_arr = np.vstack((img_arr,  img_arr_model))


        model_label_name = self.args.model +'\n'+ self.args.type
        if self.args.gan_framework == 'pix2pix':
            model_label_name = 'pix2pix\n'+model_label_name
            

        model_labels.append(model_label_name)

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
        model_name = self.args.model+'_'+ self.args.type
        plot_save_path = os.path.join(self.args.train_path,'qualitative eval')
        plot_name = model_name+ 'pred_examples.png'
        plt.savefig(os.path.join(plot_save_path,plot_name), bbox_inches='tight')



        

    









