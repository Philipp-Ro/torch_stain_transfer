import os 
import plot_utils
from pathlib import Path
import argparse

def my_args():
    parser = argparse.ArgumentParser()
    # Model
    # architectures:
    # - U_Net ==> type : S /S+att /M /M+att /L /L+att
    # - ViT ==> type : S  /M
    # - Swin ==> type : S
    # - Diffusion
    # - Classifier
    
    parser.add_argument('--model', type=str, default="", help='model architecture')
    parser.add_argument('--type', type=str, default="", help='scope of the model S or M or L')
    parser.add_argument('--pix2pix', action='store_true', default=False, help='use the generator model in gan framework')
    parser.add_argument('--score_gan', action='store_true', default=False, help='use the generator model in score gan framework')
    parser.add_argument('--diff_noise_steps', type=int, default=1000, help='Image size')

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




# -----------------------------------------------------------------------------------------------------------------
# execute main.py()
# -----------------------------------------------------------------------------------------------------------------
# get args
args = my_args()
args.img_resize = True
 # models
model_list = []
model_list.append('U-Net/3step_16f')
model_list.append('U-Net/4step_32f')
model_list.append('U-Net/5step_64f')
model_list.append('ViT/1_block_2head')
model_list.append('ViT/2_block_4head')
#model_list.append('Swin_T/2_stages_32_hidden_dim')
model_list.append('Pix2Pix/U-Net/5step_64f')
plot_utils.save_plot_for_models(args=args, model_list=model_list, IHC_score='0')