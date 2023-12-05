import os 
import plot_utils
from pathlib import Path
import argparse

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
    parser.add_argument('--gan_framework', action='store_true', default=False, help='use the generator model in gan framework')
    parser.add_argument('--diff_model', action='store_true', default=False, help='use diffusion model')
    parser.add_argument('--diff_noise_steps', type=int, default=1000, help='Image size')

    # Optimizer
    parser.add_argument('--lr', type=float, default=3e-5, help='learining rate')
    parser.add_argument('--beta1', type=float, default=0.5 , help='beta1 for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam optimizer')

    # training 
    parser.add_argument('--img_size', type=int, default=1024, help='Image size')
    parser.add_argument('--img_resize', action='store_true', default=True, help='resize image to 256')
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
    
    # Testing 
    parser.add_argument('--test_only', action='store_true', default=False, help='flag for only test')
    parser.add_argument('--num_test_epochs', type=int, default=16, help='number of test epochswith img_size=256 choose 16 for all patches in test images')
    parser.add_argument('--testplot_idx', type=list, default=[12, 18,32,115,180], help='idx for test plots in list')


    return parser.parse_args() 



# -----------------------------------------------------------------------------------------------------------------
# execute main.py()
# -----------------------------------------------------------------------------------------------------------------
# get args
args = my_args()
args.img_resize = True
 # models
model_architectures = {}
#model_architectures["U_Net"] = ['S','S+att','M','L']
#model_architectures["ViT"] = ['S','M']
model_architectures["ViT"] = ['S']
model_architectures["Swin"]= ['S']
model_architectures["U_Net"] = ['L']
gan_flag = True

# choose plots
images = {}
images["img_name"]=['01269_train_0.png','01298_train_1+.png','00917_train_2+.png','00842_train_3+.png']
#images["patch_num"]= [0,10,0,0]

# choose plot name 
plot_name = 'Pix2Pix'

# save_path
model_dir = os.path.join(Path.cwd(),"masterthesis_results")
save_path = os.path.join(model_dir,plot_name)
plot_utils.get_publishing_plot(args,images,model_architectures,gan_flag,save_path,plot_name)