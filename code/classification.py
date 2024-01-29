from torchvision.models import resnet50, ResNet50_Weights 
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch
import new_loader
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os 
from pathlib import Path
import utils
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import ViTForImageClassification
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd




class score_classifier:
    def __init__(self, args):
        super(score_classifier, self).__init__()    
        self.args = args
        self.args.img_size =1024

        # init classifier 
        model_name_or_path = 'google/vit-base-patch16-224-in21k'
        net = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=4)
        self.classifier = net.to(args.device)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.transform_resize = T.Resize((224,224))

    def fit_classifier(self,train_name):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))


        print('---------------------------------------------- ')
        print('START TRAINING CLASSIFIER')

        for epoch in range(10):
            print('---------------------------------------------- ')

            # Train data loader
            train_data = new_loader.stain_transfer_dataset( img_patch=0, set='train',args = self.args) 
            train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False) 

            
            # -----------------------------------------------------------------------------------------------------------------
            # TRAIN LOOP 1 EPOCH
            # -----------------------------------------------------------------------------------------------------------------
            for i, (real_HE, real_IHC,img_name) in enumerate(train_data_loader) :
                real_HE = self.transform_resize(real_HE)
                real_IHC = self.transform_resize(real_IHC)
                # get score 
                score = utils.get_IHC_score(img_name)

                # print progress
                if (i+1) % 600 == 0:
                    show_epoch  = 'Epoch: '+str(epoch)+'/'+str(50) +' | ' +str(i)+'/3396'
                    print(show_epoch)

                # one hot encode gt
                gt = torch.nn.functional.one_hot(score, num_classes=4)
                gt = gt.type(torch.DoubleTensor) 
          
                gt = gt.to(self.args.device)

                # predict score 
                outputs = self.classifier(real_IHC)
                outputs = outputs.logits
     
                outputs = torch.squeeze(outputs)

                # calculate loss and backprop
                loss = self.criterion(outputs, gt)     
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        checkpoint_name = 'classifier_weights'+train_name+'.pth'
        torch.save(self.classifier.state_dict(),os.path.join(Path.cwd(),checkpoint_name ) )

    def eval_classifier(self,test_name):
            test_data = new_loader.stain_transfer_dataset( img_patch=0, set='test',args = self.args) 
            test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

            best_model_weights = os.path.join(Path.cwd(),'classifier_weights'+test_name+'.pth')
            self.classifier.load_state_dict(torch.load(best_model_weights))
            predictions = []
            score_list = []

            for epoch in range(1):
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

                    fake_IHC = []
                    outputs = self.classifier(real_IHC)
                    outputs = outputs.logits

                    outputs = torch.squeeze(outputs)
                    value, idx = torch.max(outputs, 0)

                    predictions.append(idx.item())
                    score_list.append(score.item())

            cm_display = metrics.ConfusionMatrixDisplay.from_predictions(y_true=np.array(score_list), y_pred=np.array(predictions), display_labels = ["score:0", "score:1+", "score:2+", "score:3+"],cmap=plt.cm.Blues,colorbar=False)
            
            class_names = ["score:0", "score:1+", "score:2+", "score:3+"]
            report = metrics.classification_report(np.array(score_list), np.array(predictions), target_names=class_names, output_dict=True)
            df = pd.DataFrame(report).transpose()
        
            classification_report_name  = "classification_matrix.txt"
            save_path_report =os.path.join(Path.cwd(),classification_report_name)

            with open(save_path_report, 'a') as f:
                df_string = df.to_string()
                f.write(df_string)

            save_path_conf_mat =os.path.join(Path.cwd(),'ConfusionMatrix_VIT')
            # close file
            f.close()
            fig = plt.figure()
            cm_display.plot(cmap=plt.cm.Blues)
            plt.savefig(save_path_conf_mat,dpi=300)