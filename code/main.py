import argparse
import os
from architectures.U_net_Generator_model import U_net_Generator
from architectures.ViT_model import ViT_Generator
from architectures.SwinTransformer_model import SwinTransformer
from architectures.Unet_diff import UNet
from pathlib import Path
import pickle
import torch
import Trainer
import eval

# testing imports
import new_loader
from torch.utils.data import DataLoader
def my_args():
    parser = argparse.ArgumentParser()
    # Model
    # architectures:
    # - U_Net
    # - ViT
    # - Swin
    # - Diffusion
    # - Resnet
    parser.add_argument('--model', type=str, default="", help='model architecture')
    parser.add_argument('--type', type=str, default="", help='scope of the model S or M or L')
    parser.add_argument('--attention', action='store_true', default=False, help='add attention (only U_Net)')
    #parser.add_argument('--load_weights', action='store_true', default=False, help='load weights for this model')
    parser.add_argument('--gan_framework', action='store_true', default=False, help='use the generator model in gan framework')
    parser.add_argument('--diff_model', action='store_true', default=False, help='use diffusion model')

    # Optimizer
    parser.add_argument('--lr', type=float, default=3e-5, help='learining rate')
    parser.add_argument('--beta1', type=float, default=0.5 , help='beta1 for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam optimizer')

    # training 
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--img_transforms', type=list, default=[], help='choose image transforms from normalize,colorjitter,horizontal_flip,grayscale')
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
    
    # Testing 
    parser.add_argument('--num_test_epochs', type=int, default=16, help='number of test epochswith img_size=256 choose 16 for all patches in test images')
    parser.add_argument('--testplot_idx', type=list, default=[12], help='idx for test plots in list')


    return parser.parse_args() 


def load_model(args):
    if args.model == "U_Net":
        if args.type =="S":
            features= 16
            steps = 3
            model_name = "U-Net/3step_16f"

        if args.type =="M":
            features= 32
            steps = 4
            model_name = "U-Net/4step_32f"

        if args.type =="L":
            features= 64
            steps = 5
            model_name = "U-Net/5step_64f"
        
        if args.attention:
            model = model_name+'+att'

        model = U_net_Generator( in_channels=args.in_channels , out_channels=3, features=features, steps=steps, attention=args.attention)

    if args.model == "ViT":
        if args.type =="S":
            num_blocks= 1
            num_heads = 2
            model_name = "ViT/1_block_2head"
          
        if args.type =="M":
            num_blocks =2
            num_heads = 4
            model_name = "ViT/2_block_4head"

        if args.type =="L":
            num_blocks =3
            num_heads = 8
            model_name = "ViT/3_block_8head"

        model = ViT_Generator(  chw = [args.in_channels, args.img_size, args.img_size],
                            patch_size = [4,4],
                            num_heads = num_heads, 
                            num_blocks = num_blocks,
                            attention_dropout = 0.1, 
                            dropout= 0.2,
                            mlp_ratio=4
                            )
        
    if args.model == "Swin":
        if args.type =="S":
            hidden_dim = 32
            layers = [2,2]
            heads =[3, 6]
            model_name = "Swin_T/2_tages_32_hidden_dim"

        if args.type =="M":
            layers = [2,2,6]
            hidden_dim = 64
            heads =[3, 6,12]
            model_name = "Swin_T/3_tages_64_hidden_dim"

        if args.type =="L":
            layers = [2,2,6,2]
            hidden_dim = 96
            heads =[3,6,12,24]
            model_name = "Swin_T/4_tages_96_hidden_dim"

        model = SwinTransformer(    hidden_dim=hidden_dim, 
                                layers=layers, 
                                heads=heads, 
                                in_channels=args.in_channels, 
                                out_channels=3, 
                                head_dim=2, 
                                window_size=4,
                                downscaling_factors=[1, 1, 1, 1], 
                                relative_pos_embedding=True
                                )
        
    if args.model == "diff_U_Net":
        model = UNet()
        model_name = "diff_U_Net"

    # add aditional Loss to modelname
    if args.gaus_loss:
        model_name = model_name + '_gaus'

    if args.ssim_loss:
        model_name = model_name + '_ssim'

    if args.hist_loss:
        model_name = model_name + '_hist'

    # add Pix2Pix framwework to modelname
    if args.gan_framework:
        print('gan used')
        model_name = 'Pix2Pix/' + model_name

    if args.diff_model:
        print('diffusionn model')
        model_name = 'diffusion_model/' + model_name


    return model , model_name

# get args
args = my_args()
for i in vars(args):
    print(i,":",getattr(args,i))

# init model
model, model_name = load_model(args)
print(model_name)

# train model
training= Trainer.train_loop( args, model, model_name)
training.fit()

# test model
model_testing = eval.test_network(args, model, model_name)
model_testing.eval()
