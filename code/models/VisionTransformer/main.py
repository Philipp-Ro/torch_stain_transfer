import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)
import utils
import eval
import torch
import Framework_ViT
import time
from pathlib import Path
from stacked_ViT import ViT_Generator
# --------------------------- load Parameters from config ----------------------------------
config_path = os.path.join(Path.cwd(),'code\\models\\VisionTransformer\\config.yaml')
params = utils.get_config_from_yaml(config_path)

# --------------------------- intitialise cycle_Gan ----------------------------------------
model = Framework_ViT.model(params=params)
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

# ------------------------------------------------------------------------------------------
# Testing 
# ------------------------------------------------------------------------------------------

model = ViT_Generator(  chw = [params['in_channels']]+params['img_size'], 
                        patch_size = params['patch_size'],
                        num_heads = params['num_heads'], 
                        num_blocks = params['num_blocks'],
                        attention_dropout = params['attention_dropout'], 
                        dropout= params['dropout'],
                        mlp_ratio=params['mlp_ratio']
                        ).to(params['device'])
        

model_testing = eval.test_network(model,params,training_time)
model_testing.fit()