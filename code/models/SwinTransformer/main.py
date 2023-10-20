import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)
import utils
import eval
import torch
import Framework_SwinTransformer
import time
from pathlib import Path
from SwinTransformer_model import SwinTransformer
train = True
test = True
training_time = 0
# --------------------------- load Parameters from config ----------------------------------
config_path = os.path.join(Path.cwd(),'code\\models\\SwinTransformer\\config.yaml')
params = utils.get_config_from_yaml(config_path)

if train == True:
    # --------------------------- intitialise swin transformer ----------------------------------------
    model = SwinTransformer(    hidden_dim=params['hidden_dim'], 
                                layers=params['layers'], 
                                heads=params['heads'], 
                                in_channels=params['in_channels'], 
                                out_channels=params['out_channels'], 
                                head_dim=params['head_dim'], 
                                window_size=params['window_size'],
                                downscaling_factors=params['downscaling_factors'], 
                                relative_pos_embedding=params['relative_pos_embedding']
                                )
    if params['trained_model_path']!= "None":
        model.load_state_dict(torch.load(params['trained_model_path']))
        
    model = Framework_SwinTransformer.model(params=params, net=model)
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
    model = SwinTransformer( hidden_dim=params['hidden_dim'], 
                                        layers=params['layers'], 
                                        heads=params['heads'], 
                                        in_channels=params['in_channels'], 
                                        out_channels=params['out_channels'], 
                                        head_dim=params['head_dim'], 
                                        window_size=params['window_size'],
                                        downscaling_factors=params['downscaling_factors'], 
                                        relative_pos_embedding=params['relative_pos_embedding']
                                        ).to(params['device'])

    model_testing = eval.test_network(model,params,training_time).to(params['device'])
    model_testing.eval()