import os
import torchvision
import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class stain_transfer_dataset(Dataset):
    def __init__(self,img_patch, HE_img_dir,IHC_img_dir, norm=True,grayscale=False, img_size=(1024,1024)):
        self.HE_img_dir = HE_img_dir
        self.IHC_img_dir = IHC_img_dir
        self.norm = norm
        self.grayscale = grayscale
        self.transform_gray = transforms.Grayscale(3)

        if img_size[0]==img_size[1]:
            self.img_size = img_size[0]
        else:
            print('IMAGE SIZE MUST BE SQUARE')

        self.img_patch = img_patch
        

    def __len__(self):
        lst = os.listdir(self.HE_img_dir)
        self.img_names = lst
        return len(lst)


    def __getitem__(self, idx):
        HE_img_path = os.path.join(self.HE_img_dir, self.img_names[idx])
        IHC_img_path = os.path.join(self.IHC_img_dir, self.img_names[idx])

        HE_img = load_image_to_tensor(img_path=HE_img_path,norm_token = False)
        IHC_img = load_image_to_tensor(img_path=IHC_img_path,norm_token = False)

        HE_img_norm = load_image_to_tensor(img_path=HE_img_path,norm_token = True)
        IHC_img_norm = load_image_to_tensor(img_path=IHC_img_path,norm_token = True)

        HE_patches_norm = HE_img_norm.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)
        IHC_patches_norm = IHC_img_norm.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)

        HE_patches = HE_img.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)
        IHC_patches = IHC_img.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)

        # reshape the images 
        HE_tensor_norm = reshape_img(HE_patches_norm, self.img_size, self.img_patch)
        IHC_tensor_norm = reshape_img(IHC_patches_norm, self.img_size, self.img_patch)
        HE_tensor = reshape_img(HE_patches, self.img_size, self.img_patch)
        IHC_tensor = reshape_img(IHC_patches, self.img_size, self.img_patch)
       

        if self.grayscale == True:
                HE_tensor_norm = self.transform_gray(HE_tensor_norm)
                IHC_tensor_norm = self.transform_gray(IHC_tensor_norm)
                HE_tensor = self.transform_gray(HE_tensor)
                IHC_tensor = self.transform_gray(IHC_tensor)

        return HE_tensor, HE_tensor_norm, IHC_tensor, IHC_tensor_norm, self.img_names[idx]


def load_image_to_tensor(img_path,norm_token):
    pil_img = Image.open(img_path)
    
    if norm_token==True:
        transform = transforms.Compose([
        transforms.ToTensor(),
        
    ])
    else:
        transform = transforms.Compose([
        transforms.ToTensor()
    ])
        
    img_tensor = transform(pil_img)
    img_tensor = img_tensor.cuda()
    return img_tensor

def reshape_img(img, img_size, img_patch):
    num_patches = (1024 * 1024) // img_size**2 
    img = img.reshape(3,num_patches,img_size,img_size)
    img = torch.permute(img,(1,0,2,3)) 
    img = img[img_patch,:,:,:]
    return img