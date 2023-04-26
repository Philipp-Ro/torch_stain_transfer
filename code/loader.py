import os
import torchvision
import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as fn

class stain_transfer_dataset(Dataset):
    def __init__(self,img_patch, HE_img_dir,IHC_img_dir,params):
        self.HE_img_dir = HE_img_dir
        self.IHC_img_dir = IHC_img_dir
        self.img_names =  os.listdir(self.HE_img_dir)
        self.params = params

        if params['img_size'][0]==params['img_size'][1]:
            self.img_size = params['img_size'][0]
        else:
            print('IMAGE SIZE MUST BE SQUARE')

        self.img_patch = img_patch
        

    def __len__(self):
        lst = os.listdir(self.HE_img_dir)
        self.img_names = lst
        return len(lst)


    def __getitem__(self, idx):
        HE_img= load_image_to_tensor(self,idx= idx, folder_dir=self.HE_img_dir, img_names=self.img_names, preprocess_list=self.params['preprocess_HE'], img_class='HE')
        IHC_img = load_image_to_tensor(self,idx= idx, folder_dir=self.IHC_img_dir, img_names=self.img_names, preprocess_list=self.params['preprocess_IHC'], img_class='IHC')

        HE_img = HE_img.cuda()
        IHC_img = IHC_img.cuda()
     
        HE_patches = HE_img.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)
        IHC_patches = IHC_img.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)

        # reshape the images 
        HE_tensor = reshape_img(HE_patches, self.img_size, self.img_patch)
        IHC_tensor = reshape_img(IHC_patches, self.img_size, self.img_patch)
       
        return HE_tensor,  IHC_tensor, self.img_names[idx]


def load_image_to_tensor(self,idx, folder_dir, img_names, preprocess_list, img_class):
    img_path = os.path.join(folder_dir, img_names[idx])
    pil_img = Image.open(img_path)
    #img_tensor = fn.to_tensor(pil_img)
    img_tensor, skip_token = preprocess_img(self,preprocess_list, pil_img, img_class)
  
    if skip_token==1:
        img_path = os.path.join(folder_dir, img_names[idx+1])
        pil_img = Image.open(img_path)
        img_tensor = fn.to_tensor(pil_img)
        img_tensor, skip_token = preprocess_img(preprocess_list, img_tensor, img_class)

    return img_tensor


def reshape_img(img, img_size, img_patch):
    num_patches = (1024 * 1024) // img_size**2 
    img = img.reshape(3,num_patches,img_size,img_size)
    img = torch.permute(img,(1,0,2,3)) 
    img = img[img_patch,:,:,:]
    return img

def preprocess_img(self,preprocess_list, img_tensor, img_class):

    transform_list = []
    skip_token = 0
    seed = np.random.randint(2147483647) 
    random.seed(seed) 
    torch.manual_seed(seed)
    transform_list.append(transforms.ToTensor())
    # check the preprocess list and add the different transforms 
    if "normalise" in preprocess_list:
        if self.params['normalise_token'] == 'img_net':
            mean = self.params['mean_img_net']
            std = self.params['std_img_net']
        else:

            if img_class == 'HE':
            # the mean and std where calculated from the train data of the bci data set 
            # to gain specific mean and std for immunohistological images 
                mean = self.params['mean_HE']
                std = self.params['std_HE']

                transform_list.append(transforms.Normalize(mean, std))

            if  img_class == 'IHC':
            # the mean and std where calculated from the train data of the bci data set 
            # to gain specific mean and std for immunohistological images 
                mean = self.params['mean_IHC']
                std = self.params['std_IHC']

                transform_list.append(transforms.Normalize(mean, std))

    if "colorjitter" in preprocess_list:
        transform_list.append(transforms.ColorJitter(brightness=(0.7,1), contrast = (1,3), saturation=(0.7,1.3), hue=(-0.1,0.1)))

    if "grayscale" in preprocess_list:
         transform_list.append(transforms.Grayscale(3))

    # catch 2 exeptions if normalisation does not work because of std = 0 we skip the messurement
    preprocess_img = transforms.Compose(transform_list)
    img_preprocessed = preprocess_img(img_tensor)
    #if not transform_list or skip_token==1:
    #    img_preprocessed = preprocess_img(img_tensor)
    #else:
    #    preprocess_img = transforms.Compose(transform_list)
    #    img_preprocessed = preprocess_img(img_tensor)

    return img_preprocessed, skip_token



    

