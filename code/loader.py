import os
import torchvision
import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class stain_transfer_dataset(Dataset):
    def __init__(self,epoch, num_epochs, HE_img_dir,IHC_img_dir, norm=True, img_size=(1024,1024)):
        self.HE_img_dir = HE_img_dir
        self.IHC_img_dir = IHC_img_dir
        self.norm = norm
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

        if self.norm == True:
            HE_img_norm = normalise_img(self,idx,HE_img)
            IHC_img_norm = normalise_img(self,idx,IHC_img)
        else: 
            HE_img_norm = HE_img
            IHC_img_norm = IHC_img
            
        # unfold the image where Kernel_size = stride 
        # patches = img_norm.unfold(1, size, stride).unfold(2, size, stride)
        HE_patches = HE_img_norm.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)
        IHC_patches = IHC_img_norm.unfold(1, self.img_size, self.img_size).unfold(2, self.img_size, self.img_size)

        # reshape the images 
        num_patches = (1024 * 1024) // self.img_size**2 
        HE_patches = HE_patches.reshape(3,num_patches,self.img_size,self.img_size)
        IHC_patches = IHC_patches.reshape(3,num_patches,self.img_size,self.img_size)

        HE_patches = torch.permute(HE_patches,(1,0,2,3))
        IHC_patches = torch.permute(IHC_patches,(1,0,2,3))

        HE_tensor = HE_patches[self.epoch,:,:,:]
        IHC_tensor = IHC_patches[self.epoch,:,:,:]

        #HE_tensor = HE_tensor.unsqueeze(0)
        #IHC_tensor = IHC_tensor.unsqueeze(0)

        return HE_tensor,IHC_tensor


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

def normalise_img(self,idx,img):
    mean, std = img.mean([1,2]), img.std([1,2])
    if sum(std != 0) and sum(mean !=0) :
        normalise_img = transforms.Normalize(mean,std)
        img_norm = normalise_img(img)
    else:
        
        std=[0.001,0.001,0.001]
    
    normalise_img = transforms.Normalize(mean,std)
    img_norm = normalise_img(img)
    return img_norm