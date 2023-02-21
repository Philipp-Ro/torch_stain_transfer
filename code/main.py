import torch
import matplotlib.pyplot as plt
import numpy as np
import utils
import conv_models
import os , itertools
import CycleNet




print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))





params = utils.get_config_from_yaml('C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/torch_stain_transfer/code/config.yaml')
######################## initialise models ############################################
gen_G = conv_models.GeneratorResNet(3, num_residual_blocks=9)
gen_F = conv_models.GeneratorResNet(3, num_residual_blocks=9)

disc_X = conv_models.Discriminator(3)
disc_Y = conv_models.Discriminator(3)

######################### set models for cuda #########################################
gen_G = gen_G.cuda()
gen_F = gen_F.cuda()

disc_X = disc_X.cuda()
disc_Y = disc_Y.cuda()

######################### initialize weights ##########################################
gen_G.apply(CycleNet.weights_init_normal)
gen_F.apply(CycleNet.weights_init_normal)
disc_X.apply(CycleNet.weights_init_normal)
disc_Y.apply(CycleNet.weights_init_normal)

######################## initialize optimizier ########################################
gen_optimizer = torch.optim.Adam(itertools.chain(gen_G.parameters(), gen_F.parameters()), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))
disc_X_optimizer = torch.optim.Adam(disc_X.parameters(), lr=params['learn_rate_disc'], betas=(params['beta1'], params['beta2']))
disc_Y_optimizer = torch.optim.Adam(disc_Y.parameters(), lr=params['learn_rate_disc'], betas=(params['beta1'], params['beta2']))

######################## initialize CycleGan ##########################################
model = CycleNet.model(params,gen_G, gen_F,disc_X, disc_Y, disc_X_optimizer, disc_Y_optimizer, gen_optimizer)

######################## train model ##################################################
gen_G = model.fit()