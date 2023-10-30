import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)
import utils
import eval
import torch
import Framework_U_net
import time
from pathlib import Path
from BCI_UNet import UnetGenerator
from Resnet_gen import ResnetGenerator
from U_net_Generator_model import U_net_Generator
import matplotlib.pyplot as plt
import pickle

train = True
test = True
training_time = 0
# --------------------------- load Parameters from config ----------------------------------
config_path = os.path.join(Path.cwd(),'code\\models\\simple_U_net\\config.yaml')
params = utils.get_config_from_yaml(config_path)
output_folder_path = os.path.join(params['output_path'],params['output_folder'])

if train == True:
    # ------------------------------------------------------------------------------------------------
    # train model 
    # ------------------------------------------------------------------------------------------------
    #
    # --------------------------- intitialise model  -------------------------------------------
    if params['gen_architecture']== "Resnet":
        num_features = 64
        model = ResnetGenerator(input_nc=params['in_channels'], output_nc=3, ngf=num_features, n_blocks=9).to(params['device'])

    if params['gen_architecture']== "BCI_UNet":
        model = UnetGenerator(input_nc=params['in_channels'], output_nc=3, num_downs=params['num_steps'], ngf=params['num_features']).to(params['device'])

    if params['gen_architecture']== "my_Unet":
        model = U_net_Generator( in_channels=params['in_channels'], out_channels=3, features=params['num_features'], steps=params['num_steps'], attention=params['attention']).to(params['device'])
        
    
    # --------------------------- load weights and train results  -------------------------------------
    train_eval ={}
    if params['trained_model_dir']!= "None":
        train_result_dir_load = os.path.join(params['trained_model_dir'],'train_result')
        model.load_state_dict(torch.load(os.path.join(params['trained_model_dir'],'gen_G_weights_final.pth')))
        print('model loaded')
        with open(train_result_dir_load, "rb") as fp:   # Unpickling
            load_data_train = pickle.load(fp)
        train_eval['mse'] = load_data_train['mse']
        train_eval['ssim'] = load_data_train['ssim']
        train_epoch_name = 'epoch_'+str(len(train_eval['mse']))+'_to_'+str(len(train_eval['mse'])+params['num_epochs'])

    else:
        train_eval['mse'] = []
        train_eval['ssim'] = []
        train_epoch_name = 'epoch_'+str(1)+'_to_'+str(len(train_eval['mse'])+params['num_epochs'])

    # ---------------------------- get path folder ---------------------------------------------------------
    epoch_path = os.path.join(output_folder_path,train_epoch_name)
    os.mkdir(epoch_path)

    # ---------------------------- save config --------------------------------------------------------------
    config_save_path =  os.path.join(epoch_path,'config.yaml')
    utils.save_config_in_dir(config_save_path, params)
    
    # ---------------------------- init train loop ----------------------------------------------------------
    model = Framework_U_net.model(params=params, net=model,epoch_path=epoch_path,train_eval=train_eval)

    # --------------------------- Train Network -------------------------------------------------------------
    start = time.time()
    gen, train_eval  = model.fit()
    stop = time.time()

    # --------------------------- save Network ---------------------------------------------------------------
    training_time = (stop-start)/60
    
    model_path = os.path.join(epoch_path,params['model_name'])
    torch.save(gen.state_dict(), model_path)

    # --------------------------- plot train results -----------------------------------------------------------
    x = range(len(train_eval['mse']))

    fig, axs = plt.subplots(2)
    fig.suptitle('train_results')
    axs[0].plot(x, train_eval['mse'])
    axs[0].set_title('MSE')
    axs[1].plot(x, train_eval['ssim'])
    axs[1].set_title('SSIM')

    fig.savefig(os.path.join(epoch_path,"train_result.png"))

if test == True:
    # ------------------------------------------------------------------------------------------
    # Testing 
    # ------------------------------------------------------------------------------------------
    if params['gen_architecture']== "Resnet":
        num_features = 64
        model = ResnetGenerator(input_nc=params['in_channels'], output_nc=3, ngf=num_features, n_blocks=9).to(params['device'])
    if params['gen_architecture']== "BCI_UNet":
        num_steps = 8
        num_features = 64
        model= UnetGenerator(input_nc=params['in_channels'], output_nc=3, num_downs=num_steps, ngf=num_features).to(params['device'])
    if params['gen_architecture']== "my_Unet":
        model = U_net_Generator( in_channels=params['in_channels'], out_channels=3, features=params['num_features'], steps=params['num_steps'], attention=params['attention']).to(params['device'])

    model_testing = eval.test_network(model,params,training_time,epoch_path)
    model_testing.eval()