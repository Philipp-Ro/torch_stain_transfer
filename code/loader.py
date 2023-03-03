import os
import torchvision
import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class stain_transfer_dataset(Dataset):
    def __init__(self,epoch, num_epochs, HE_img_dir,IHC_img_dir, norm=True,grayscale=False, img_size=(1024,1024)):
        self.HE_img_dir = HE_img_dir
        self.IHC_img_dir = IHC_img_dir
        self.norm = norm
        self.grayscale = grayscale
        self.transform_gray = transforms.Grayscale(3)

        if img_size[0]==img_size[1]:
            self.img_size = img_size[0]
        else:
            print('IMAGE SIZE MUST BE SQUARE')

        self.epoch = epoch
        self.num_epochs = num_epochs

    def __len__(self):
        lst = os.listdir(self.HE_img_dir)
        self.img_names = lst
        return len(lst)


    def __getitem__(self, idx):
        HE_img_path = os.path.join(self.HE_img_dir, self.img_names[idx])
        IHC_img_path = os.path.join(self.IHC_img_dir, self.img_names[idx])

        HE_img = load_image_to_tensor(HE_img_path)
        IHC_img = load_image_to_tensor(IHC_img_path)

        
        HE_img_norm = normalise_img(HE_img)
        IHC_img_norm = normalise_img(IHC_img)
       
             
        # unfold the image where Kernel_size = stride 
        # patches = img_norm.unfold(1, size, stride).unfold(2, size, stride)
        HE_patches_norm = HE_img_norm.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)
        IHC_patches_norm = IHC_img_norm.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)

        HE_patches = HE_img.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)
        IHC_patches = IHC_img.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)


        # reshape the images 
        HE_tensor_norm = reshape_img(HE_patches_norm, self.img_size, self.epoch)
        IHC_tensor_norm = reshape_img(IHC_patches_norm, self.img_size, self.epoch)
        HE_tensor = reshape_img(HE_patches, self.img_size, self.epoch)
        IHC_tensor = reshape_img(IHC_patches, self.img_size, self.epoch)

        if self.grayscale == True:
                HE_tensor_norm = self.transform_gray(HE_tensor_norm)
                IHC_tensor_norm = self.transform_gray(IHC_tensor_norm)
                HE_tensor = self.transform_gray(HE_tensor)
                IHC_tensor = self.transform_gray(IHC_tensor)

        return HE_tensor, HE_tensor_norm, IHC_tensor, IHC_tensor_norm


def load_image_to_tensor(img_path):
    pil_img = Image.open(img_path)
    img_arr = np.array(pil_img)
    # type conversion
    img_arr = img_arr.astype(np.float32)
    # rearange dims for tensor 
    img_arr = np.transpose(img_arr, axes=[2,0,1])

    # get tensor 
    img_tensor = torch.tensor(img_arr)

    img_tensor = img_tensor.cuda()
    return img_tensor

def normalise_img(img):
    mean, std = img.mean([1,2]), img.std([1,2])
    if sum(std != 0) and sum(mean !=0) :
        normalise_img = transforms.Normalize(mean,std)
        img_norm = normalise_img(img)
    else:
        
        std=[0.001,0.001,0.001]
    
    normalise_img = transforms.Normalize(mean,std)
    img_norm = normalise_img(img)
    return img_norm

def reshape_img(img, img_size, epoch):
    num_patches = (1024 * 1024) // img_size**2 
    img = img.reshape(3,num_patches,img_size,img_size)
    img = torch.permute(img,(1,0,2,3)) 
    img = img[epoch,:,:,:]
    return img