import torch
params = {
    'batch_size':1,
    'input_size':256,
    'resize_scale':286,
    'crop_size':256,
    'fliplr':True,
    #model params
    'num_epochs':100,
    'decay_epoch':100,
    'ngf':32,   #number of generator filters
    'ndf':64,   #number of discriminator filters
    'num_resnet':6, #number of resnet blocks
    'lrG':0.0002,    #learning rate for generator
    'lrD':0.0002,    #learning rate for discriminator
    'beta1':0.5 ,    #beta1 for Adam optimizer
    'beta2':0.999 ,  #beta2 for Adam optimizer
    'lambdaA':10 ,   #lambdaA for cycle loss
    'lambdaB':10  ,  #lambdaB for cycle loss
}


class CycleNet(torch.nn.Module):
    def __init__(self, generator_G, generator_F, discriminator_X, discriminator_Y, l_cycle, l_ssim, l_id):
        super(CycleGan, self).__init__()
        # initialse the generator G and F
        # gen_F transfers from doman Y -> X
        # gen_G transfers from doman X -> Y
        # 
        # initialize the discriminator X and Y
        # disc_X distinguishes between real and fake in the X domain 
        # disc_Y distinguishes between real and fake in the Y domain 
        #
        #
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = l_cycle
        self.lambda_id = l_id
        self.lambda_ssim = l_ssim  

        # weigths of models
        self.gen_G.normal_weight_init(mean=0.0, std=0.02)
        self.gen_F.normal_weight_init(mean=0.0, std=0.02)
        self.disc_X.normal_weight_init(mean=0.0, std=0.02)
        self.disc_Y.normal_weight_init(mean=0.0, std=0.02)

    def train_model(self,params):
        for epoch in range(params['num_epochs']):
            disc_X_losses = []
            disc_Y_losses = []
            gen_G_losses = []
            gen_F_losses = []
            cycle_A_losses = []
            cycle_B_losses = []

            # -------------------------- train generator G --------------------------
            # X --> Y
            # HE --> IHC
            fake_IHC = self.gen_G(real_HE)
            disc_Y_fake_decision = self.disc_Y(fake_IHC)
            gen_G_loss = MSE_Loss(disc_Y_fake_decision, Variable(torch.ones(disc_Y_fake_decision.size()).cuda()))

            # -------------------------- train generator F -------------------------- 
            # Y --> X
            # HE --> IHC
            fake_HE = self.gen_F(real_IHC)
            disc_X_fake_decision = self.disc_X(fake_HE)
            gen_F_loss = MSE_Loss(D_A_fake_decision, Variable(torch.ones(disc_X_real_decision.size()).cuda()))


            # ------------------------- forward cycle loss --------------------------
            recon_HE = self.gen_F(fake_IHC)
            cycle_A_loss = L1_Loss(recon_HE, real_HE) * params['forward_cycle_löambda']
                
                
            # ---------------------------backward cycle loss-------------------------
            recon_IHC = self.gen_G(fake_HE)
            cycle_B_loss = L1_Loss(recon_IHC, real_IHC) * params['backward_cycle_lambda']
                
            # ------------------------- Back propagation genarator ------------------
            gen_total_loss = (gen_G_loss + gen_F_loss + cycle_A_loss + cycle_B_loss) * params['generator_lambda']
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # -------------------------- train discriminator D_A --------------------------
            disc_X_real_decision = self.disc_X(real_HE)
            # calculation loss of fake disc x  and an 
            disc_X_real_loss = MSE_Loss(disc_X_real_decision, Variable(torch.ones(disc_X_real_decision.size()).cuda()))
            
            
            
            disc_X_fake_decision = self.disc_X(fake_HE)
            disc_X_fake_loss = MSE_Loss(disc_X_fake_decision, Variable(torch.zeros(disc_X_fake_decision.size()).cuda()))
            
            # Back propagation
            disc_X_loss = (disc_X_real_loss + disc_X_fake_loss) * params['disc_lambda']
            disc_X_optimizer.zero_grad()
            disc_X_loss.backward()
            disc_X_optimizer.step()
            
            # -------------------------- train discriminator D_B --------------------------
            disc_Y_real_decision = self.disc_Y(real_IHC)
            disc_Y_real_loss = MSE_Loss(disc_Y_real_decision, Variable(torch.ones(disc_Y_fake_decision.size()).cuda()))
            
            
            
            disc_Y_fake_decision = self.disc_Y(fake_IHC)
            disc_Y_fake_loss = MSE_Loss(disc_Y_fake_decision, Variable(torch.zeros(disc_Y_fake_decision.size()).cuda()))
            
            # Back propagation
            disc_Y_loss = (disc_Y_real_loss + disc_Y_fake_loss) * params['disc_lambda']
            disc_Y_optimizer.zero_grad()
            disc_Y_loss.backward()
            disc_Y_optimizer.step()
            
            # ------------------------ Print -----------------------------
            # loss values
            disc_X_losses.append(disc_X_loss.item())
            disc_Y_losses.append(disc_Y_loss.item())
            gen_G_losses.append(gen_G_loss.item())
            gen_F_losses.append(gen_F_loss.item())
            cycle_A_losses.append(cycle_A_loss.item())
            cycle_B_losses.append(cycle_B_loss.item())

