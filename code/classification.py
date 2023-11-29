from torchvision.models import resnet50, ResNet50_Weights 
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


class score_classifier:
    def __init__(self, args):
        super(score_classifier, self).__init__()    
        self.args = args
        self.args.img_size =1024

        # init classifier 
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  
        net.fc = torch.nn.Linear(net.fc.in_features, 4)
        torch.nn.init.xavier_uniform_(net.fc.weight)
        self.classifier = net.to(args.device)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.transform_resize = T.Resize((256,256))

    def fit_classifier(self,train_name):
        params_1x = [param for name, param in self.classifier.named_parameters() if 'fc' not in str(name)]
        optimizer = torch.optim.Adam([{'params':params_1x}, {'params': self.classifier.fc.parameters(), 'lr': self.args.lr*10}], lr=self.args.lr, weight_decay=5e-4)

        print('---------------------------------------------- ')
        print('START TRAINING')

        for epoch in range(50):
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
                if img_name[0].endswith("0.png"):
                    score = torch.tensor(0)

                if img_name[0].endswith("1+.png"):
                    score = torch.tensor(1)

                if img_name[0].endswith("2+.png"):
                    score = torch.tensor(2)

                if img_name[0].endswith("3+.png"):
                    score =torch.tensor(3)

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
                outputs = torch.squeeze(outputs)

                # calculate loss and backprop
                loss = self.criterion(outputs, gt)     
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        checkpoint_name = 'classifier_weights'+train_name+'.pth'
        torch.save(self.classifier.state_dict(),os.path.join(Path.cwd(),checkpoint_name ) )

    def test_classifier(self, classifier_name,model, model_name):

        if self.args.model != "None":
            model = model.to(self.args.device)
            result_dir = os.path.join(Path.cwd(),"masterthesis_results")
            self.train_path = os.path.join(result_dir,model_name)
            best_model_weights = os.path.join(self.train_path,'final_weights_gen.pth')
            model.load_state_dict(torch.load(best_model_weights))
        else:
            self.train_path = Path.cwd()


        best_model_weights = os.path.join(Path.cwd(),'classifier_weights'+classifier_name+'.pth')
        self.classifier.load_state_dict(torch.load(best_model_weights))
        predictions = []
        score_list = []

        for epoch in range(1):
            test_data = new_loader.stain_transfer_dataset( img_patch=0, set='test',args = self.args) 
            test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 
            for i, (real_HE, real_IHC,img_name) in enumerate(test_data_loader) :

                
                real_HE = self.transform_resize(real_HE)
                real_IHC = self.transform_resize(real_IHC)

                if img_name[0].endswith("0.png"):
                    score = torch.tensor(0)

                if img_name[0].endswith("1+.png"):
                    score = torch.tensor(1)

                if img_name[0].endswith("2+.png"):
                    score = torch.tensor(2)  

                if img_name[0].endswith("3+.png"):
                    score =torch.tensor(3)

                if self.args.model != "None":
                    outputs = self.classifier(model(real_IHC))
                else:
                    outputs = self.classifier(real_IHC)

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
        save_path_report =os.path.join(self.train_path,classification_report_name)

        with open(save_path_report, 'a') as f:
            df_string = df.to_string()
            f.write(df_string)



        # close file
        f.close()
        fig = plt.figure()
        cm_display.plot(cmap=plt.cm.Blues)
        conf_mat_name = 'conf_mat'+classifier_name+'.png'
        save_path_conf_mat =os.path.join(self.train_path,conf_mat_name)
        plt.savefig(save_path_conf_mat,dpi=300)