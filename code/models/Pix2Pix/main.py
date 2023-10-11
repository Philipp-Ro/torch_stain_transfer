import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)
import utils
import eval
import torch
import Framework_Pix2Pix
import time
from pathlib import Path
from U_net_model import UNet
from BCI_UNet import UnetGenerator
from ViT_model import ViT_Generator
from Resnet_gen import ResnetGenerator

train = True
test = True
training_time = 0

# --------------------------- load Parameters from config ----------------------------------
config_path = os.path.join(Path.cwd(),'code\\models\\Pix2Pix\\config.yaml')
params = utils.get_config_from_yaml(config_path)

if train == True:
    

    # --------------------------- intitialise cycle_Gan ----------------------------------------
    if params['gen_architecture']== "Resnet":
        num_features = 64
        gen = ResnetGenerator(input_nc=params['in_channels'], output_nc=3, ngf=num_features, n_blocks=9).to(params['device'])

    if params['gen_architecture']== "BCI_UNet":
        num_steps = 8
        num_features = 64
        gen = UnetGenerator(input_nc=params['in_channels'], output_nc=3, num_downs=num_steps, ngf=num_features).to(params['device'])

    if params['gen_architecture']== "Unet":
        gen = UNet(in_channels=params['in_channels'],out_channels=3, init_features=params['gen_features']).to(params['device'])

    if params['gen_architecture']== "transformer":
        gen = ViT_Generator(   chw = [params['in_channels']]+params['img_size'], 
                                    patch_size = params['patch_size'],
                                    num_heads = params['num_heads'], 
                                    num_blocks = params['num_blocks'],
                                    attention_dropout = params['attention_dropout'], 
                                    dropout= params['dropout'],
                                    mlp_ratio=params['mlp_ratio']
                                    ).to(params['device'])
    if params['trained_model_path']!= "None":
        gen.load_state_dict(torch.load(params['trained_model_path']))

    model = Framework_Pix2Pix.model(params=params,gen=gen)
    # --------------------------- Train Network ------------------------------------------------
    start = time.time()
    gen = model.fit()
    stop = time.time()

    # ------------------------------------------------------------------------------------------
    # save the trained model 
    # ------------------------------------------------------------------------------------------
    training_time = (stop-start)/60
    output_folder_path = os.path.join(params['output_path'],params['output_folder'])
    model_path = os.path.join(output_folder_path,params['model_name'])
    config_path =  os.path.join(output_folder_path,'config.yaml')

    utils.save_config_in_dir(config_path, params)
    torch.save(gen.state_dict(), model_path)

if test == True:
# ------------------------------------------------------------------------------------------
# Testing 
# ------------------------------------------------------------------------------------------
    # model =  UNet(in_channels=params['in_channels'],out_channels=3, init_features=params['gen_features']).to(params['device'])
    if params['gen_architecture']== "Resnet":
        num_features = 64
        model = ResnetGenerator(input_nc=params['in_channels'], output_nc=3, ngf=num_features, n_blocks=9).to(params['device'])

    if params['gen_architecture']== "BCI_UNet":
        num_steps = 8
        num_features = 64
        model = UnetGenerator(input_nc=params['in_channels'], output_nc=3, num_downs=num_steps, ngf=num_features).to(params['device'])

    if params['gen_architecture']== "Unet":
        gen = UNet(in_channels=params['in_channels'],out_channels=3, init_features=params['gen_features']).to(params['device'])

    if params['gen_architecture']== "transformer":
        gen = ViT_Generator(   chw = [params['in_channels']]+params['img_size'], 
                                    patch_size = params['patch_size'],
                                    num_heads = params['num_heads'], 
                                    num_blocks = params['num_blocks'],
                                    attention_dropout = params['attention_dropout'], 
                                    dropout= params['dropout'],
                                    mlp_ratio=params['mlp_ratio']
                                    ).to(params['device'])
        
    if params['trained_model_path']!= "None":
        model.load_state_dict(torch.load(params['trained_model_path']))

        
    model_testing = eval.test_network(model,params,training_time)
    model_testing.eval()