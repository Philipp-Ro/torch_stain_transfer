import sys
import torch
import os
from torch.utils.data import Dataset, DataLoader
import loader
import matplotlib.pyplot as plt
import numpy as np

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))


train_data = loader.stain_transfer_dataset('C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/Data_set_BCI_challange/train/HE_imgs/HE',
                                           'C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/Data_set_BCI_challange/train/IHC_imgs/IHC',
                                           img_size=(100,100))

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=False)
HE_img_batch, IHC_img_batch  = next(iter(train_dataloader))
HE_img = HE_img_batch[0].squeeze()
IHC_img = IHC_img_batch[0].squeeze()

fig, (HE_ax, IHC_ax) = plt.subplots(2)
HE_ax.imshow(np.transpose(HE_img))
IHC_ax.imshow(np.transpose(IHC_img))
plt.show()