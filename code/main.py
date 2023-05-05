import torch
import utils
import conv_models
import os , itertools
import CycleNet
import eval
import trans_models
# ------------------------------------------------------------------------------------------
# load config and intialize Generators and Discriminators
# ------------------------------------------------------------------------------------------

params = utils.get_config_from_yaml('C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/torch_stain_transfer/code/config.yaml')


in_channels = 3

if params['gen_architecture'] == 'conv':
    
    gen_G = conv_models.Generator(in_channels= params['in_channels'],
                                        num_residual_blocks = params['num_resnet'],
                                        U_net_filter_groth = params['U_net_filter_groth'],
                                        U_net_step_num = params['U_net_step_num']
                                        )
    
    gen_F = conv_models.Generator(in_channels= params['in_channels'], 
                                        num_residual_blocks = params['num_resnet'],
                                        U_net_filter_groth = params['U_net_filter_groth'],
                                        U_net_step_num = params['U_net_step_num']
                                        )
    

    disc_X = conv_models.Discriminator(in_channels= in_channels)
    disc_Y = conv_models.Discriminator(in_channels= in_channels)

    gen_G = gen_G.cuda()
    gen_F = gen_F.cuda()

    disc_X = disc_X.cuda()
    disc_Y = disc_Y.cuda()

    gen_G.apply(CycleNet.weights_init_normal)
    gen_F.apply(CycleNet.weights_init_normal)
    disc_X.apply(CycleNet.weights_init_normal)
    disc_Y.apply(CycleNet.weights_init_normal)


if params['gen_architecture']== 'trans':
    
    gen_G = trans_models.Generator(
                                      chw = [3,256,256], 
                                      patch_size = [32, 32],
                                      num_heads = 2, 
                                      num_blocks = 2)
    
    
    gen_F =  trans_models.Generator(
                                      chw = [3,256,256], 
                                      patch_size = [32, 32],
                                      num_heads = 2, 
                                      num_blocks = 2)
    
    disc_X =  trans_models.Discriminator(
                                      chw = [3,256,256], 
                                      patch_size = [32, 32],
                                      num_heads = 2, 
                                      num_blocks = 2)
    
    disc_Y =  trans_models.Discriminator(
                                      chw = [3,256,256], 
                                      patch_size = [32, 32],
                                      num_heads = 2, 
                                      num_blocks = 2)

    gen_G = gen_G.cuda()
    gen_F = gen_F.cuda()

    disc_X = disc_X.cuda()
    disc_Y = disc_Y.cuda()  

# ------------------------------------------------------------------------------------------
# intitialise optimisers and Cyclenet
# ------------------------------------------------------------------------------------------
gen_optimizer = torch.optim.Adam(itertools.chain(gen_G.parameters(), gen_F.parameters()), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))

disc_optimizer = torch.optim.Adam(itertools.chain(disc_X.parameters(), disc_Y.parameters()), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))


model = CycleNet.model(params,gen_G, gen_F,disc_X, disc_Y, disc_optimizer, gen_optimizer)

# --------------------------- Train Network ------------------------------------------------
gen_G, gen_F, disc_X, disc_Y = model.fit()

# ------------------------------------------------------------------------------------------
# save the trained model 
# ------------------------------------------------------------------------------------------
output_folder_path = os.path.join(params['output_path'],params['output_folder'])
model_path = os.path.join(output_folder_path,params['model_name'])
config_path =  os.path.join(output_folder_path,'config.yaml')

utils.save_config_in_dir(config_path, params)
torch.save(gen_G.state_dict(), model_path)

# ------------------------------------------------------------------------------------------
# Testing 
# ------------------------------------------------------------------------------------------
model_testing = eval.test_network(params)
model_testing.fit()