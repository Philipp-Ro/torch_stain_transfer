import os
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from random import randrange
class stain_transfer_dataset(Dataset):
    def __init__(self,HE_img_dir,IHC_img_dir, transform=None, img_size=(1024,1024)):
        self.HE_img_dir = HE_img_dir
        self.IHC_img_dir = IHC_img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        lst = os.listdir(self.IHC_img_dir)
        self.img_names = lst
        return len(lst)


    def __getitem__(self, idx):
        HE_img_path = os.path.join(self.HE_img_dir, self.img_names[idx])
        IHC_img_path = os.path.join(self.IHC_img_dir, self.img_names[idx])

        HE_img = read_image(HE_img_path)
        IHC_img = read_image(IHC_img_path)

        kernel_size, stride = self.img_size

        HE_patches = HE_img.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        HE_patches = HE_patches.contiguous().view(HE_patches.size(0), -1, kernel_size, kernel_size)

        IHC_patches = IHC_img.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        IHC_patches = IHC_patches.contiguous().view(IHC_patches.size(0), -1, kernel_size, kernel_size)

        num_patches = len(HE_patches[1,:,1,1])
        current_patch_num = randrange(num_patches)
        
        return HE_patches[:,current_patch_num,:,:], IHC_patches[:,current_patch_num,:,:]

