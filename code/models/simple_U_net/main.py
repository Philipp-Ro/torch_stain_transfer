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

from U_net_model import UNet

# --------------------------- load Parameters from config ----------------------------------
config_path = os.path.join(Path.cwd(),'code\\models\\simple_U_net\\config.yaml')
params = utils.get_config_from_yaml(config_path)

# --------------------------- intitialise cycle_Gan ----------------------------------------
model = Framework_U_net.model(params=params)
# --------------------------- Train Network ------------------------------------------------
start = time.time()
gen, train_eval = model.fit()
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
model = UNet(in_channels=params['in_channels'],out_channels=3, init_features=32).to(params['device'])
#model = U_net_Generator(in_channels=params['in_channels'], features=params['gen_features']).to(params['device'])

model_testing = eval.test_network(model,params,training_time)
model_testing.fit()