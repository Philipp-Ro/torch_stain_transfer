import torch
import matplotlib.pyplot as plt
import numpy as np
import utils
import conv_models
import os , itertools
import CycleNet
import transformer_models
import loader 

from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio




params = utils.get_config_from_yaml('C:/Users/phili/OneDrive/Uni/WS_22/Masterarbeit/Masterarbeit_Code_Philipp_Rosin/torch_stain_transfer/code/config.yaml')


if params['conv_net'] == 1:
    
    gen_G = conv_models.GeneratorResNet(3, num_residual_blocks=9)
    gen_F = conv_models.GeneratorResNet(3, num_residual_blocks=9)
    gen_test = conv_models.GeneratorResNet(3, num_residual_blocks=9)

    disc_X = conv_models.Discriminator(3)
    disc_Y = conv_models.Discriminator(3)

    gen_G = gen_G.cuda()
    gen_F = gen_F.cuda()
    gen_test = gen_test.cuda()

    disc_X = disc_X.cuda()
    disc_Y = disc_Y.cuda()

    gen_G.apply(CycleNet.weights_init_normal)
    gen_F.apply(CycleNet.weights_init_normal)
    disc_X.apply(CycleNet.weights_init_normal)
    disc_Y.apply(CycleNet.weights_init_normal)


if params['trans_net']== 1:
    
    gen_G = transformer_models.Generator(   img_size= params['img_size'][0],
                                            embedding_dim=0,
                                            patch_size=params['patch_size'],
                                            in_channels=params['in_channels'],
                                            dropout_embedding=params['dropout_embedding'],
                                            nhead= params['nhead'],
                                            num_layers=params['num_layers']
                                            )
    
    gen_F = transformer_models.Generator(   img_size= params['img_size'][0],
                                            embedding_dim=0,
                                            patch_size=params['patch_size'],
                                            in_channels=params['in_channels'],
                                            dropout_embedding=params['dropout_embedding'],
                                            nhead= params['nhead'],
                                            num_layers=params['num_layers']
                                            )
    
    gen_test = transformer_models.Generator(   img_size= params['img_size'][0],
                                            embedding_dim=0,
                                            patch_size=params['patch_size'],
                                            in_channels=params['in_channels'],
                                            dropout_embedding=params['dropout_embedding'],
                                            nhead= params['nhead'],
                                            num_layers=params['num_layers']
                                            )
    
    disc_X = conv_models.Discriminator(3)
    disc_Y = conv_models.Discriminator(3)

    gen_G = gen_G.cuda()
    gen_F = gen_F.cuda()
    gen_test = gen_test.cuda()

    disc_X = disc_X.cuda()
    disc_Y = disc_Y.cuda()  


######################## initialize optimizier ########################################
gen_optimizer = torch.optim.Adam(itertools.chain(gen_G.parameters(), gen_F.parameters()), lr=params['learn_rate_gen'], betas=(params['beta1'], params['beta2']))
disc_X_optimizer = torch.optim.Adam(disc_X.parameters(), lr=params['learn_rate_disc'], betas=(params['beta1'], params['beta2']))
disc_Y_optimizer = torch.optim.Adam(disc_Y.parameters(), lr=params['learn_rate_disc'], betas=(params['beta1'], params['beta2']))

######################## initialize CycleGan ##########################################
model = CycleNet.model(params,gen_G, gen_F,disc_X, disc_Y, disc_X_optimizer, disc_Y_optimizer, gen_optimizer)

######################## train model ##################################################
gen_G, gen_F, disc_X, disc_Y = model.fit()


######################## save model and config ###########################################
output_folder_path = "{}{}".format(params['output_path'],params['output_folder'])
gen_G_path = "{}{}".format(output_folder_path,'/Generator_G_weights.pth')
config_path =  "{}{}".format(output_folder_path,'/config.yaml')

utils.save_config_in_dir(config_path, params)
torch.save(gen_G.state_dict(), gen_G_path)



####################################### testing ###########################################################
model = gen_test
model.load_state_dict(torch.load(gen_G_path))
model.eval()

# set up result vector 
result = {}
result['epoch'] = []
result['ssim_mean'] = []
result['ssim_std'] = []
result['psnr_mean'] = []
result['psnr_std'] = []

# set up test data dirs 
test_path = params['test_dir']
HE_img_dir = "{}{}".format(test_path,'/HE_imgs/HE')
IHC_img_dir = "{}{}".format(test_path,'/IHC_imgs/IHC')

for epoch in range(params['num_epochs']):
    
    result['epoch'].append(epoch)
    test_data = loader.stain_transfer_dataset(  epoch = epoch,
                                                num_epochs = params['num_epochs'],
                                                HE_img_dir = HE_img_dir,
                                                IHC_img_dir = IHC_img_dir,
                                                img_size= params['img_size'],
                                                )
    
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = ssim.cuda()
    ssim_scores = []

    psnr = PeakSignalNoiseRatio()
    psnr = psnr.cuda()
    psnr_scores = []

    for i, (real_HE, real_IHC) in enumerate(test_data_loader):
        fake_IHC = model(real_HE)
        
        ssim_scores.append(ssim(fake_IHC, real_IHC))
        psnr_scores.append(psnr(fake_IHC, real_IHC))
        torch.cuda.empty_cache()

    result['ssim_mean'].append(np.mean(ssim_scores))
    result['ssim_std'].append(np.std(ssim_scores))

    result['psnr_mean'].append(np.mean(psnr_scores))
    result['psnr_std'].append(np.std(psnr_scores))

# open file for writing
f = open(output_folder_path,"w")

# write file
f.write( str(result) )

# close file
f.close()