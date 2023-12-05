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
                outputs = torch.squeeze(outputs)

                # calculate loss and backprop
                loss = self.criterion(outputs, gt)     
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        checkpoint_name = 'classifier_weights'+train_name+'.pth'
        torch.save(self.classifier.state_dict(),os.path.join(Path.cwd(),checkpoint_name ) )

   
   