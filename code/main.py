import argparse
import plot_utils
import Trainer
import testing
import utils
import os
from pathlib import Path
import time
import classification
import torch
# -----------------------------------------------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------------------------------------------
def my_args():
    parser = argparse.ArgumentParser()
    # Model
    # architectures:
    # - U_Net ==> type : S /S+att /M /M+att /L /L+att
    # - ViT ==> type : S  /M
    # - Swin ==> type : S
    # - Diffusion
    # - None
    
    parser.add_argument('--model', type=str, default="", help='model architecture')
    parser.add_argument('--type', type=str, default="", help='scope of the model S or M or L')
    #parser.add_argument('--attention', action='store_true', default=False, help='add attention (only U_Net)')
    parser.add_argument('--gan_framework', action='store_true', default=False, help='use the generator model in gan framework')
    parser.add_argument('--diff_model', action='store_true', default=False, help='use diffusion model')
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
    
    # Testing 
    parser.add_argument('--test_only', action='store_true', default=False, help='flag for only test')
    parser.add_argument('--num_test_epochs', type=int, default=16, help='number of test epochswith img_size=256 choose 16 for all patches in test images')
    parser.add_argument('--testplot_idx', type=list, default=[12, 18,32,115,180], help='idx for test plots in list')

    parser.add_argument('--classifier_only', action='store_true', default=False, help='flag for only classifer')


    return parser.parse_args() 



# -----------------------------------------------------------------------------------------------------------------
# execute main.py()
# -----------------------------------------------------------------------------------------------------------------
# get args
args = my_args()
for i in vars(args):
    print(i,":",getattr(args,i))

# init model
model, model_name = utils.load_model(args)
print(model_name)
train_time = 0
# train model
if not args.test_only and not args.classifier_only :
    start = time.time()
    training= Trainer.train_loop( args, model, model_name)
    training.fit()
    end = time.time()
    train_time = end-start

# test model
if not args.classifier_only:
    if "diff" not in model_name:
        model_testing = testing.test_network(args, model, model_name)
        model_testing.get_full_eval( 'test', model, group_wise=True, train_time=train_time )
        model_testing.get_full_eval( 'train', model, group_wise=True, train_time=train_time )
    else:
        result_dir = os.path.join(Path.cwd(),"masterthesis_results")
        train_path = os.path.join(result_dir,model_name)
        best_model_weights = os.path.join(train_path,'final_weights_gen.pth')
        model.load_state_dict(torch.load(best_model_weights))
        print(' ---------------------------------------------- ')
        print('weights loaded')
        model_testing = testing.test_network(args, model, model_name)
        model_testing.get_full_eval( 'test', model, group_wise=True, train_time=train_time )
        model_testing.get_full_eval( 'train', model, group_wise=True, train_time=train_time )



classifier_test = classification.score_classifier(args)
classifier_test.test_classifier("_IHC_256_50",model, model_name)


   
