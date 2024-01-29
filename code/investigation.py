import argparse
import new_loader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
def my_args():
    parser = argparse.ArgumentParser()
    #
    # models and types:
    # - U_Net ==> type : S /S+att /M /M+att /L /L+att
    # - ViT ==> type : S  /M
    # - Swin ==> type : S
    # - Diffusion
    # - Classifier
    #
    # gan_framework:
    # - pix2pix
    # - score_gan
    
    parser.add_argument('--model', type=str, default="", help='model architecture')
    parser.add_argument('--type', type=str, default="", help='scope of the model S or M or L')
    #parser.add_argument('--pix2pix', action='store_true', default=False, help='use the generator model in gan framework')
    #parser.add_argument('--score_gan', action='store_true', default=False, help='use the generator model in score gan framework')
    parser.add_argument('--diff_noise_steps', type=int, default=1000, help='Image size')
    parser.add_argument('--gan_framework', type=str, default="None", help='set a gan framework')

    # Optimizer
    parser.add_argument('--lr', type=float, default=3e-5, help='learining rate')
    parser.add_argument('--beta1', type=float, default=0.5 , help='beta1 for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam optimizer')

    # training 
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--img_resize', action='store_true', default=False, help='resize image to 256')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--img_transforms', type=list, default=["colorjitter",'horizontal_flip','vertical_flip'], help='choose image transforms from normalize,colorjitter,horizontal_flip,grayscale')
    parser.add_argument('--num_epochs', type=int, default=100, help='epoch num')
    parser.add_argument('--decay_epoch', type=int, default=80, help='decay epoch num')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--device', type=str, default="cuda", help='device')


    # Loss
    parser.add_argument('--gaus_loss', action='store_true', default=False, help='activate gausian blurr loss')
    parser.add_argument('--ssim_loss', action='store_true', default=False, help='activate ssim  loss')
    parser.add_argument('--hist_loss', action='store_true', default=False, help='activate histogram loss')

    # Data dirs
    parser.add_argument('--train_data', type=str, default='C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/Data_set_BCI_challange/train', help='directory to the train data')
    parser.add_argument('--test_data', type=str, default='C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/Data_set_BCI_challange/val', help='directory to the test data')
    
    parser.add_argument('--train_path', type=str, default='', help='directory to the training')
    parser.add_argument('--train_eval_path', type=str, default='', help='directory to the train eval file')
    parser.add_argument('--test_eval_path', type=str, default='', help='directory to the test eval file')
    parser.add_argument('--tp_path', type=str, default='', help='directory to the train plots')
    parser.add_argument('--c_path', type=str, default='', help='directory to the checkpoints of the training')


    # Testing 
    parser.add_argument('--quant_eval_only', action='store_true', default=False, help='flag for only test')
    parser.add_argument('--qual_eval_only', action='store_true', default=False, help='flag for only classifer')


    return parser.parse_args() 
args = my_args()
loss_SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
loss_PSNR = PeakSignalNoiseRatio().to(args.device)
loss_MSE = nn.MSELoss().to(args.device)
loss_CE = nn.CrossEntropyLoss()



mse_epoch_list = []
ssim_epoch_list = []
psnr_epoch_list = []

for patch in range(16):
    print(patch)
    data_set_init = new_loader.stain_transfer_dataset( img_patch=patch, set='train', args=args) 
    loader = DataLoader(data_set_init, batch_size=1, shuffle=False) 
    mse_list = []
    ssim_list = []
    psnr_list = []

    for i, (real_HE, real_IHC, img_name) in enumerate(loader):
        mse_list.append(loss_MSE(real_HE,real_IHC).item())
        ssim_list.append(loss_SSIM(real_HE,real_IHC).item())
        psnr_list.append(loss_PSNR(real_HE,real_IHC).item())
    
    mse_epoch_list.append(np.mean(mse_list))
    ssim_epoch_list.append(np.mean(ssim_list))
    psnr_epoch_list.append(np.mean(psnr_list))


print(np.mean(mse_epoch_list))
print(np.mean(ssim_epoch_list))
print(np.mean(psnr_epoch_list))

for patch in range(16):
    print(patch)
    #for patch in range(1):
    data_set_init = new_loader.stain_transfer_dataset( img_patch=patch, set='test', args=args) 
    loader = DataLoader(data_set_init, batch_size=1, shuffle=False) 
    for i, (real_HE, real_IHC, img_name) in enumerate(loader):
        mse_list.append(loss_MSE(real_HE,real_IHC).item())
        ssim_list.append(loss_SSIM(real_HE,real_IHC).item())
        psnr_list.append(loss_PSNR(real_HE,real_IHC).item())

print(np.mean(mse_list))
print(np.mean(ssim_list))
print(np.mean(psnr_list))