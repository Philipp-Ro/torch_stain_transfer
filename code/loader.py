import os
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class stain_transfer_dataset(Dataset):
    def __init__(self,HE_img_dir,IHC_img_dir, transform=None):
        self.HE_img_dir = HE_img_dir
        self.IHC_img_dir = IHC_img_dir
        self.transform = transform

    def __len__(self):
        lst = os.listdir(self.IHC_img_dir)
        self.img_names = lst
        return len(lst)

    def __getitem__(self, idx):
        HE_img_path = os.path.join(self.HE_img_dir, self.img_names[idx])
        IHC_img_path = os.path.join(self.IHC_img_dir, self.img_names[idx])

        HE_img = read_image(HE_img_path)
        IHC_img = read_image(IHC_img_path)

        if self.transform:
            HE_img = self.transform(HE_img)
            IHC_img = self.transform(IHC_img)
        
        return HE_img, IHC_img 


'''
x = torch.randn(3, 24, 24) # channels, height, width
kernel_size, stride = 12, 12
patches = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)
print(patches.shape) # channels, patches, kernel_size, kernel_size
> torch.Size([3, 4, 12, 12])
'''