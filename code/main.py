import argparse
import plot_utils
import Trainer
import testing
import utils
import time
import os
import pickle
# -----------------------------------------------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------------------------------------------
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
    parser.add_argument('--num_epochs', type=int, default=2, help='epoch num')
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
for i in vars(args):
    print(i,":",getattr(args,i))

# init model
model ,model_framework, model_arch, model_specs = utils.build_model(args)

if model_framework != None:
    model_result_path = model_framework+model_arch+model_specs
else:
    model_result_path = model_arch+model_specs

args, model_name, train_plot_eval, test_plot_eval = utils.set_paths(args, model_framework, model_arch, model_specs)

train_time = 0
# train model
if not args.quant_eval_only and not args.qual_eval_only:
    start = time.time()
    training = Trainer.train_loop( args, model, train_plot_eval, test_plot_eval)
    trained_model, train_plot_eval, test_plot_eval = training.fit()
    end = time.time()
    train_time = end-start

# init evaluation
trained_model = utils.load_model_weights(args, model, model_name)

model_testing = testing.test_network(args, trained_model)

# quantitativ evaluation
if args.quant_eval_only and not args.qual_eval_only:
    quant_eval_path = os.path.join(args.train_path,'quantitative eval 2')
    if not os.path.isdir(quant_eval_path):
        result_test, test_list_mse, test_list_ssim, test_list_psnr= model_testing.get_full_quant_eval('test', trained_model, train_time=train_time)
        result_train, train_list_mse, train_list_ssim, train_list_psnr = model_testing.get_full_quant_eval('train', trained_model, train_time=train_time)
    else:
        mse_test_path = os.path.join(quant_eval_path,'mse_list_test')
        mse_train_path = os.path.join(quant_eval_path,'mse_list_train')
        with open(mse_test_path, "rb") as fp:   
            test_list_mse = pickle.load(fp)

        with open(mse_train_path, "rb") as fp:   
            train_list_mse = pickle.load(fp)

        ssim_test_path = os.path.join(quant_eval_path,'ssim_list_test')
        ssim_train_path = os.path.join(quant_eval_path,'ssim_list_train')
        with open(ssim_test_path, "rb") as fp:   
            test_list_ssim = pickle.load(fp)

        with open(ssim_train_path, "rb") as fp:   
            train_list_ssim = pickle.load(fp)

        psnr_test_path = os.path.join(quant_eval_path,'psnr_list_test')
        psnr_train_path = os.path.join(quant_eval_path,'psnr_list_train')
        with open(psnr_test_path, "rb") as fp:   
            test_list_psnr = pickle.load(fp)

        with open(psnr_train_path, "rb") as fp:   
            train_list_psnr = pickle.load(fp)



    

    latex_table_path = os.path.join(args.train_path,'quantitative eval/summary_latex.txt')

    plot_utils.get_boxplot_for_scores(args, metric_name='MSE',train_list = train_list_mse, test_list= test_list_mse, path= quant_eval_path)
    plot_utils.get_boxplot_for_scores(args, metric_name='SSIM',train_list = train_list_ssim, test_list= test_list_ssim, path= quant_eval_path)
    plot_utils.get_boxplot_for_scores(args, metric_name='PSNR',train_list = train_list_psnr, test_list= test_list_psnr, path= quant_eval_path)
    utils.write_summary_latex_table(result_test, result_train, latex_table_path)

if args.qual_eval_only and not args.quant_eval_only:
    model_list = [model_result_path]   
    images = model_testing.get_full_qual_eval(classifier_name ="ViT_resize", model=trained_model, model_list=model_list)




   
