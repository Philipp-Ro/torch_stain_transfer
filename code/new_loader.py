import os
import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
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

        # init preprossing
        self.preprocess_img = init_img_preproces(args)

    def __len__(self):
        lst = os.listdir(self.HE_img_dir)
        self.img_names = lst
        return len(lst)


    def __getitem__(self, idx):
        seed = np.random.randint(2147483647) 

        HE_img = load_img(idx= idx, folder_dir=self.HE_img_dir, img_names=self.img_names)
        IHC_img = load_img(idx= idx, folder_dir=self.IHC_img_dir, img_names=self.img_names)
        
        random.seed(seed) 
        torch.manual_seed(seed)
        HE_tensor= self.preprocess_img(HE_img)

        random.seed(seed) 
        torch.manual_seed(seed)
        IHC_tensor = self.preprocess_img(IHC_img)


        HE_patches = HE_tensor.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)
        IHC_patches = IHC_tensor.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)

        # reshape the images 
        HE_out = reshape_img(HE_patches, self.img_size, self.img_patch)
        IHC_out = reshape_img(IHC_patches, self.img_size, self.img_patch)

        HE_out = HE_out.to(self.args.device)
        IHC_out = IHC_out.to(self.args.device)

        return HE_out,  IHC_out, self.img_names[idx]


def load_img(idx, folder_dir, img_names):
    img_path = os.path.join(folder_dir, img_names[idx])
    pil_img = Image.open(img_path)

    return pil_img
    


def reshape_img(img, img_size, img_patch):
    num_patches = (1024 * 1024) // img_size**2 
    img = img.reshape(3,num_patches,img_size,img_size)
    img = torch.permute(img,(1,0,2,3)) 
    img = img[img_patch,:,:,:]
    return img




def init_img_preproces(args):
    transform_list = []

    # check the preprocess list and add the different transforms 
    if "normalise" in args.img_transforms and not args.model=='Diffusion':
        
        mean =  [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        transform_list.append(transforms.Normalize(mean, std))

    if "colorjitter" in args.img_transforms:
        transform_list.append(transforms.ColorJitter(contrast = (1,3)))

    if "grayscale" in args.img_transforms:
         transform_list.append(transforms.Grayscale(3))

    if "horizontal_flip"in args.img_transforms:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if "vertical_flip"in args.img_transforms:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    if args.diff_model:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform_list.append(transforms.ToTensor())
    preprocess_img = transforms.Compose(transform_list)

    return preprocess_img



    

