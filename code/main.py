import torch
import utils
import conv_models
import os , itertools
import CycleNet
import transformer_models
import eval

# -----------------------------------------------------------------------------------------
# load config and intialize Generators and Discriminators
# ------------------------------------------------------------------------------------------

params = utils.get_config_from_yaml('C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/torch_stain_transfer/code/config.yaml')

# set in_channels of networks depending on grayscale 
if params['grayscale'] == True:
    in_channels = 3 # grayscale == 1 ? 
else:
     in_channels = 3

if params['gen_architecture'] == 'conv':
    
    gen_G = conv_models.GeneratorResNet(in_channels= params['in_channels'],
                                        num_residual_blocks = params['num_resnet']
                                        )
    
    gen_F = conv_models.GeneratorResNet(in_channels= params['in_channels'], 
                                        num_residual_blocks = params['num_resnet']
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
    
    gen_G = transformer_models.Generator(   img_size= params['img_size'][0],
                                            embedding_dim=0,
                                            patch_size=params['patch_size'],
                                            in_channels=in_channels,
                                            dropout_embedding=params['dropout_embedding'],
                                            nhead= params['nhead'],
                                            num_layers=params['num_layers']
                                            )
    
    gen_F = transformer_models.Generator(   img_size= params['img_size'][0],
                                            embedding_dim=0,
                                            patch_size=params['patch_size'],
                                            in_channels=in_channels,
                                            dropout_embedding=params['dropout_embedding'],
                                            nhead= params['nhead'],
                                            num_layers=params['num_layers']
                                            )
    
    
    disc_X = conv_models.Discriminator(in_channels= params['in_channels'])
    disc_Y = conv_models.Discriminator(in_channels= params['in_channels'])

    gen_G = gen_G.cuda()
    gen_F = gen_F.cuda()

    disc_X = disc_X.cuda()
    disc_Y = disc_Y.cuda()  

# -----------------------------------------------------------------------------------------
# intitialise optimisers and Cyclenet
# ------------------------------------------------------------------------------------------
gen_optimizer = torch.optim.Adam(itertools.chain(gen_G.parameters(), gen_F.parameters()), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))

disc_optimizer = torch.optim.Adam(itertools.chain(disc_X.parameters(), disc_Y.parameters()), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))


model = CycleNet.model(params,gen_G, gen_F,disc_X, disc_Y, disc_optimizer, gen_optimizer)
# train network
gen_G, gen_F, disc_X, disc_Y = model.fit()


# -----------------------------------------------------------------------------------------
# save the trained model 
# ------------------------------------------------------------------------------------------
output_folder_path = os.path.join(params['output_path'],params['output_folder'])
model_path = os.path.join(output_folder_path,params['model_name'])
config_path =  os.path.join(output_folder_path,'config.yaml')

utils.save_config_in_dir(config_path, params)
torch.save(gen_G.state_dict(), model_path)

# -----------------------------------------------------------------------------------------
# Testing 
# ------------------------------------------------------------------------------------------

model_testing = eval.test_network(params)
model_testing.fit()