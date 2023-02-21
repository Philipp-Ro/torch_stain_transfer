import os
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from random import randrange
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class stain_transfer_dataset(Dataset):
    def __init__(self,epoch, num_epochs, HE_img_dir,IHC_img_dir, transform=None, img_size=(1024,1024)):
        self.HE_img_dir = HE_img_dir
        self.IHC_img_dir = IHC_img_dir
        self.transform = transform
        self.img_size = img_size
        self.epoch = epoch
        self.num_epochs = num_epochs

    def __len__(self):
        lst = os.listdir(self.IHC_img_dir)
        self.img_names = lst
        return len(lst)


    def __getitem__(self, idx):
        HE_img_path = os.path.join(self.HE_img_dir, self.img_names[idx])
        IHC_img_path = os.path.join(self.IHC_img_dir, self.img_names[idx])

        HE_img_orig  = Image.open(HE_img_path)
        IHC_img_orig = Image.open(IHC_img_path)

        img_height, img_width = self.img_size

        if img_height != img_width:
            print('CHOOSE IMAGESIZE WHERE HIGHT == WIDTH')
        else:
            # implement a loop which iterates through patches per epoch 
            patch_num = self.epoch % self.num_epochs

            HE_img = get_img_crop(HE_img_orig,patch_num, img_height,img_width)
            IHC_img = get_img_crop(IHC_img_orig,patch_num, img_height,img_width)

            convert_tensor = transforms.ToTensor()

            HE_tensor = convert_tensor(HE_img)
            IHC_tensor = convert_tensor(IHC_img)

            HE_tensor = HE_tensor.cuda()
            IHC_tensor = IHC_tensor.cuda()

        return HE_tensor,IHC_tensor


# original images gets cropped 
def get_img_crop(img_orig,patch_num, img_height,img_width):
    source_hight, source_width = img_orig.size

    num_patches_row = round(source_width/img_width)
    
    row_num = round(patch_num/num_patches_row)
    col_num = patch_num % num_patches_row

    left = (col_num*img_width)-1
    right = left+img_width
    top = (row_num*img_height)-1
    bottom = top+img_height

    cropped_img = img_orig.crop((left, top, right, bottom))

    return cropped_img