import torch
import utils
import conv_models
import os , itertools
import Cycle_Gan_Net
import eval
import trans_models
import Gan_Net
# ------------------------------------------------------------------------------------------
# load config and intialize Generators 
# ------------------------------------------------------------------------------------------

params = utils.get_config_from_yaml('C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/torch_stain_transfer/code/config.yaml')

if params['gen_architecture'] == 'conv':
    
    gen_G = conv_models.Generator(in_channels= params['in_channels'],
                                        num_residual_blocks = params['num_resnet'],
                                        hidden_dim= params['hidden_dim'],
                                        U_net_filter_groth = params['U_net_filter_groth'],
                                        U_net_step_num = params['U_net_step_num']
                                        )
    
    gen_F = conv_models.Generator(in_channels= params['in_channels'], 
                                        num_residual_blocks = params['num_resnet'],
                                        hidden_dim= params['hidden_dim'],
                                        U_net_filter_groth = params['U_net_filter_groth'],
                                        U_net_step_num = params['U_net_step_num']
                                        )
    
    gen_G = gen_G.cuda()
    gen_F = gen_F.cuda()

    gen_G.apply(Cycle_Gan_Net.weights_init_normal)
    gen_F.apply(Cycle_Gan_Net.weights_init_normal)



if params['gen_architecture']== 'trans':
    gen_G = trans_models.Generator(
                                      chw = [params['in_channels']]+params['img_size'], 
                                      patch_size = params['patch_size_gen'],
                                      num_heads = params['num_heads_gen'], 
                                      num_blocks = params['num_blocks_gen'])
    
    
    gen_F =  trans_models.Generator(
                                      chw = [params['in_channels']]+params['img_size'], 
                                      patch_size = params['patch_size_gen'],
                                      num_heads = params['num_heads_gen'], 
                                      num_blocks = params['num_blocks_gen'])
    
    
    gen_G = gen_G.cuda()
    gen_F = gen_F.cuda()

    gen_G = trans_models.init_weights(gen_G)
    gen_F = trans_models.init_weights(gen_F)

# ------------------------------------------------------------------------------------------
# load config and intialize Discriminators
# ------------------------------------------------------------------------------------------ 
if params['disc_architecture'] == 'conv':
    img_shape = params['img_size']
    img_shape.append(params['in_channels'])
    disc_X = conv_models.Discriminator(
                                        disc_step_num=params['disc_step_num'],
                                        disc_filter_groth=params['disc_filter_groth'],
                                        in_channels= params['in_channels'],
                                        hidden_dim=params['hidden_dim'])
                                
                                       
                                       
    
    disc_Y = conv_models.Discriminator(
                                        disc_step_num=params['disc_step_num'],
                                        disc_filter_groth=params['disc_filter_groth'],
                                        in_channels= params['in_channels'],
                                        hidden_dim=params['hidden_dim'])
                                       
                                       

    disc_X = disc_X.cuda()
    disc_Y = disc_Y.cuda()

    disc_X.apply(Gan_Net.weights_init_normal)
    disc_Y.apply(Gan_Net.weights_init_normal)

if params['disc_architecture']== 'trans':
    disc_X =  trans_models.Discriminator(
                                      chw = [params['in_channels']]+params['img_size'], 
                                      patch_size = params['patch_size_disc'],
                                      num_heads = params['num_heads_disc'], 
                                      num_blocks = params['num_blocks_disc'])
    
    disc_Y =  trans_models.Discriminator(
                                      chw = [params['in_channels']]+params['img_size'], 
                                      patch_size = params['patch_size_disc'],
                                      num_heads = params['num_heads_disc'], 
                                      num_blocks = params['num_blocks_disc'])
    
    disc_X = disc_X.cuda()
    disc_Y = disc_Y.cuda() 

    disc_X = trans_models.init_weights(disc_X)
    disc_Y = trans_models.init_weights(disc_Y)

# ------------------------------------------------------------------------------------------
# intitialise frame model
# ------------------------------------------------------------------------------------------

if params['frame_architecture']== 'cycleGan':
# --------------------------- intitialise optimisers ---------------------------------------
    gen_optimizer = torch.optim.Adam(itertools.chain(gen_G.parameters(), gen_F.parameters()), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))
    disc_optimizer = torch.optim.Adam(itertools.chain(disc_X.parameters(), disc_Y.parameters()), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))

# --------------------------- intitialise cycle_Gan ----------------------------------------
    model = Cycle_Gan_Net.model(params,gen_G, gen_F,disc_X, disc_Y, disc_optimizer, gen_optimizer)
# --------------------------- Train Network ------------------------------------------------
    gen_G, gen_F, disc_X, disc_Y = model.fit()


if params['frame_architecture']== 'Gan':
# --------------------------- intitialise optimisers ---------------------------------------
    gen_optimizer = torch.optim.Adam(gen_G.parameters(), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))
    disc_optimizer = torch.optim.Adam(disc_X.parameters(), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))
    
# --------------------------- intitialise Gan ----------------------------------------------
    model = Gan_Net.model(params,gen_G,disc_X, disc_optimizer, gen_optimizer)
# --------------------------- Train Network ------------------------------------------------
    gen, disc = model.fit()


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