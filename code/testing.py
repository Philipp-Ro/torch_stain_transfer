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


class test_network():
    def __init__(self, args, model, model_name):
        self.args = args
        self.model_name = model_name

        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
        self.PSNR = PeakSignalNoiseRatio().to(args.device)
        self.MSE = nn.MSELoss().to(args.device)
        self.CE = torch.nn.CrossEntropyLoss()

        # init classifier
        classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  
        classifier.fc = torch.nn.Linear(classifier.fc.in_features, 4)
        torch.nn.init.xavier_uniform_(classifier.fc.weight)
        self.classifier = classifier.to(args.device)

        self.transform_resize = T.Resize((256,256))

        self.model = model.to(args.device)


        num_epochs = args.num_epochs

        # init test path
        testing_name = 'quant_eval_'+str(num_epochs)+'_epochs'
        self.testing_path = os.path.join(args.train_path,testing_name)
        os.mkdir(self.testing_path)

        #save model in test file 
        model_name_epoch =  'model_final_weights_'+str(num_epochs)+'_epochs.pth'
        final_model_path = os.path.join(self.testing_path,model_name_epoch)
        torch.save(model.state_dict(), final_model_path)


        if "diff" not in model_name:
            # load train data
            with open(args.train_eval_path, "rb") as fp:   
                    self.train_plot_eval = pickle.load(fp)

            with open(args.test_eval_path, "rb") as fp:   
                    self.test_plot_eval = pickle.load(fp)

            # plot the train metrics
            plot_utils.plot_trainresult(self.args, self.testing_path, self.train_plot_eval, self.test_plot_eval)
            
        

    def get_full_quant_eval(self, data_set, model, group_wise, train_time ):
        result_total = utils.init_eval()
        prediction_time = []
        result_total['train_time'] = train_time
        num_patches = ((1024 * 1024) // self.args.img_size**2)-1
        print(num_patches)
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


    def get_full_qual_eval(self, classifier_name, model, plot_names):
            img_arr = []
            # init qual eval dir
            self.save_path = os.path.join(self.args.train_path,"qulitative eval")
            os.mkdir(self.save_path)

            # set save names
            model_png_name = self.args.model +'_'+self.args.type
            best_model_weights = os.path.join(Path.cwd(),'classifier_weights'+classifier_name+'.pth')
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
                    if img_name[0].endswith("0.png"):
                        score = torch.tensor(0)

                    if img_name[0].endswith("1+.png"):
                        score = torch.tensor(1)

                    if img_name[0].endswith("2+.png"):
                        score = torch.tensor(2)  

                    if img_name[0].endswith("3+.png"):
                        score =torch.tensor(3)

                    # get IHC score prediction 
                    if self.args.model == 'Diffusion':
                        fake_IHC = diffusion.sample(model , n=real_IHC.shape[0], y=real_HE)
                        outputs = self.classifier(fake_IHC)

                    elif self.args.model == "Classifier":
                        fake_IHC = []
                        outputs = self.classifier(real_IHC)
                        
                    else:
                        fake_IHC = model(real_HE)
                        outputs = self.classifier(fake_IHC)

                    
                    outputs = torch.squeeze(outputs)
                    value, idx = torch.max(outputs, 0)

                    predictions.append(idx.item())
                    score_list.append(score.item())


            #cf_matrix = confusion_matrix(np.array(score_list), np.array(predictions))
            cm_display = metrics.ConfusionMatrixDisplay.from_predictions(y_true=np.array(score_list), y_pred=np.array(predictions), display_labels = ["score:0", "score:1+", "score:2+", "score:3+"],cmap=plt.cm.Blues,colorbar=False)
            
            class_names = ["score:0", "score:1+", "score:2+", "score:3+"]
            report = metrics.classification_report(np.array(score_list), np.array(predictions), target_names=class_names, output_dict=True)
            df = pd.DataFrame(report).transpose()
    
            classification_report_name  = "classification_matrix.txt"
            save_path_report =os.path.join(self.save_path,classification_report_name)

            with open(save_path_report, 'a') as f:
                df_string = df.to_string()
                f.write(df_string)

            # close file
            f.close()
            fig = plt.figure()
            cm_display.plot(cmap=plt.cm.Blues)
            
            conf_mat_name = model_png_name+'_conf_mat.png'
            
            if self.args.gan_framework:
                conf_mat_name = 'Pix2Pix_'+conf_mat_name

            save_path_conf_mat =os.path.join(self.save_path,conf_mat_name)
            plt.savefig(save_path_conf_mat,dpi=300)

            return img_arr

    def plot_img_set_for_net(self,images):
            
            num_rows = 3 
            num_cols = 4

            
            model_label_name = self.args.model +'\n'+self.args.type

            plot_name = self.args.model +'_'+self.args.type

            if self.args.gan_framework:
                plot_name = 'Pix2Pix_'+plot_name
                model_label_name = 'Pix2Pix\n'+model_label_name

            column_labels = ['img\ngroup 3+','img\ngroup 2+','img\ngroup 1+','img\ngroup 0']
            row_labels = [model_label_name,'IHC\nTarget','HE\nInput']

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
                plt.gcf().text(0.065, 0.2+(i*0.25), label, fontsize=18)


            plt.subplots_adjust(wspace=0, hspace=0)
            plot_name =plot_name+ '_pred_examples.png'
            plt.savefig(os.path.join(self.save_path,plot_name), bbox_inches='tight')










