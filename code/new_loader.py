import os
import torchvision
import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as fn
# -------------------------------------------------------------------------------------------------------------------------
# Loader
# -------------------------------------------------------------------------------------------------------------------------
# the loader will load the BCI images and give out a list of (real_HE, real_IHC, img_name)
# the values if NO preprocessing will have a VALUE_RANGE of [0....1]
# preprocessing options are:
# 'normalise' with eiether bci or img_net token will normalize the image data to mean()=0 and std()=1 according to the bci datast or the imagenet dataset
# 'colorjitter' will apply a colorjitter
# 'grayscale' will transform the images into black and white


class stain_transfer_dataset(Dataset):
    def __init__(self,img_patch, set,args):
        self.args = args
        if set == 'train':
            self.HE_img_dir = os.path.join(args.train_data,'HE')
            self.IHC_img_dir = os.path.join(args.train_data,'IHC')
        if set == 'test':
            self.HE_img_dir = os.path.join(args.test_data,'HE')
            self.IHC_img_dir = os.path.join(args.test_data,'IHC')

        self.img_names =  os.listdir(self.HE_img_dir)
        self.img_size = args.img_size
        self.img_patch = img_patch
        

    def __len__(self):
        lst = os.listdir(self.HE_img_dir)
        self.img_names = lst
        return len(lst)


    def __getitem__(self, idx):
        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)

        HE_img= load_image_to_tensor(self,idx= idx, folder_dir=self.HE_img_dir, img_names=self.img_names, preprocess_list=self.args.img_transforms)
        IHC_img = load_image_to_tensor(self,idx= idx, folder_dir=self.IHC_img_dir, img_names=self.img_names, preprocess_list=self.args.img_transforms)

        HE_img = HE_img.to(self.args.device)
        IHC_img = IHC_img.to(self.args.device)

     
        HE_patches = HE_img.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)
        IHC_patches = IHC_img.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)


        # reshape the images 
        HE_tensor = reshape_img(HE_patches, self.img_size, self.img_patch)
        IHC_tensor = reshape_img(IHC_patches, self.img_size, self.img_patch)
       
        return HE_tensor,  IHC_tensor, self.img_names[idx]


def load_image_to_tensor(self,idx, folder_dir, img_names, preprocess_list):
    img_path = os.path.join(folder_dir, img_names[idx])
    pil_img = Image.open(img_path)

    img_tensor = preprocess_img(self,preprocess_list, pil_img)
  


    return img_tensor


def reshape_img(img, img_size, img_patch):
    num_patches = (1024 * 1024) // img_size**2 
    img = img.reshape(3,num_patches,img_size,img_size)
    img = torch.permute(img,(1,0,2,3)) 
    img = img[img_patch,:,:,:]
    return img

def preprocess_img(self,preprocess_list, img_tensor):

    transform_list = []
    seed = np.random.randint(2147483647) 
    random.seed(seed) 
    torch.manual_seed(seed)
    # check the preprocess list and add the different transforms 
    if "normalise" in preprocess_list and not self.args.model=='Diffusion':
        
        mean =  [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        transform_list.append(transforms.Normalize(mean, std))

    if "colorjitter" in preprocess_list:
        transform_list.append(transforms.ColorJitter(brightness=(0.7,1), contrast = (1,3), saturation=(0.7,1.3), hue=(-0.1,0.1)))

    if "grayscale" in preprocess_list:
         transform_list.append(transforms.Grayscale(3))

    transform_list.append(transforms.ToTensor())

    if self.args.diff_model:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    preprocess_img = transforms.Compose(transform_list)
    img_preprocessed = preprocess_img(img_tensor)


    return img_preprocessed



    

