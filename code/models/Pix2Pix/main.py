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

# --------------------------- load Parameters from config ----------------------------------
#params = utils.get_config_from_yaml('C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/torch_stain_transfer/code/config.yaml')
config_path = os.path.join(Path.cwd(),'code\\models\\Pix2Pix\\config.yaml')
params = utils.get_config_from_yaml(config_path)

# --------------------------- intitialise cycle_Gan ----------------------------------------
model = Framework_Pix2Pix.model(params=params)
# --------------------------- Train Network ------------------------------------------------
start = time.time()
gen, disc = model.fit()
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

# ------------------------------------------------------------------------------------------
# Testing 
# ------------------------------------------------------------------------------------------
model =  UNet(in_channels=params['in_channels'],out_channels=3, init_features=params['gen_features']).to(params['device'])
model_testing = eval.test_network(model,params,training_time)
model_testing.fit()