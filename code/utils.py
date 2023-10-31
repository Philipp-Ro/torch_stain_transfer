
from torchvision import transforms
import torch 
import kornia
from architectures.U_net_Generator_model import U_net_Generator
from architectures.ViT_model import ViT_Generator
from architectures.SwinTransformer_model import SwinTransformer
from architectures.Unet_diff import UNet
from architectures.Resnet_gen import ResnetGenerator
# denormalize
def denomalise(mean,std,img):
    
    unorm = transforms.Normalize(mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
                             std=[1/std[0], 1/std[1], 1/std[2]])
    
    denomalised_img = unorm(img)
    return denomalised_img

# hist_loss
def hist_loss(self,real_img, fake_img):
    real_img = real_img.squeeze()
    fake_img = fake_img.squeeze()
    hist_loss_list = []
    for c in range(self.args.in_channels):

        hist_real = torch.histc(real_img[c,:,:], bins=64, min=-5, max=5)
        hist_real  /= hist_real .sum()

        hist_fake = torch.histc(fake_img[c,:,:], bins=64, min=-5, max=5)
        hist_fake /= hist_fake.sum()

        minima = torch.minimum(hist_real, hist_fake)
        intersection = torch.true_divide(minima.sum(), hist_fake.sum())
        hist_loss_list.append(1-intersection)

    return sum(hist_loss_list)

# gausian blurr loss 
def gausian_blurr_loss(MSE_LOSS,real_img, fake_img):
    octave1_layer2_fake=kornia.filters.gaussian_blur2d(fake_img,(3,3),(1,1))
    octave1_layer3_fake=kornia.filters.gaussian_blur2d(octave1_layer2_fake,(3,3),(1,1))
    octave1_layer4_fake=kornia.filters.gaussian_blur2d(octave1_layer3_fake,(3,3),(1,1))
    octave1_layer5_fake=kornia.filters.gaussian_blur2d(octave1_layer4_fake,(3,3),(1,1))
    octave2_layer1_fake=kornia.filters.blur_pool2d(octave1_layer5_fake, 1, stride=2)
    octave1_layer2_real=kornia.filters.gaussian_blur2d(real_img,(3,3),(1,1))
    octave1_layer3_real=kornia.filters.gaussian_blur2d(octave1_layer2_real,(3,3),(1,1))
    octave1_layer4_real=kornia.filters.gaussian_blur2d(octave1_layer3_real,(3,3),(1,1))
    octave1_layer5_real=kornia.filters.gaussian_blur2d(octave1_layer4_real,(3,3),(1,1))
    octave2_layer1_real=kornia.filters.blur_pool2d(octave1_layer5_real, 1, stride=2)
    G_L2 = MSE_LOSS(octave2_layer1_fake, octave2_layer1_real) 
    return G_L2,octave2_layer1_fake,octave2_layer1_real

def load_model(args):
    if args.model == "U_Net":
        if args.type =="S":
            features= 16
            steps = 3
            model_name = "U-Net/3step_16f"

        if args.type =="M":
            features= 32
            steps = 4
            model_name = "U-Net/4step_32f"

        if args.type =="L":
            features= 64
            steps = 5
            model_name = "U-Net/5step_64f"
        
        if args.attention:
            model = model_name+'+att'

        model = U_net_Generator( in_channels=args.in_channels , out_channels=3, features=features, steps=steps, attention=args.attention)

    if args.model == "ViT":
        if args.type =="S":
            num_blocks= 1
            num_heads = 2
            model_name = "ViT/1_block_2head"
          
        if args.type =="M":
            num_blocks =2
            num_heads = 4
            model_name = "ViT/2_block_4head"

        if args.type =="L":
            num_blocks =3
            num_heads = 8
            model_name = "ViT/3_block_8head"

        model = ViT_Generator(  chw = [args.in_channels, args.img_size, args.img_size],
                            patch_size = [4,4],
                            num_heads = num_heads, 
                            num_blocks = num_blocks,
                            attention_dropout = 0.1, 
                            dropout= 0.2,
                            mlp_ratio=4
                            )
        
    if args.model == "Swin":
        if args.type =="S":
            hidden_dim = 32
            layers = [2,2]
            heads =[3, 6]
            model_name = "Swin_T/2_stages_32_hidden_dim"

        if args.type =="M":
            layers = [2,2,6]
            hidden_dim = 64
            heads =[3, 6,12]
            model_name = "Swin_T/3_stages_64_hidden_dim"

        if args.type =="L":
            layers = [2,2,6,2]
            hidden_dim = 96
            heads =[3,6,12,24]
            model_name = "Swin_T/4_stages_96_hidden_dim"

        model = SwinTransformer(    hidden_dim=hidden_dim, 
                                layers=layers, 
                                heads=heads, 
                                in_channels=args.in_channels, 
                                out_channels=3, 
                                head_dim=2, 
                                window_size=4,
                                downscaling_factors=[1, 1, 1, 1], 
                                relative_pos_embedding=True
                                )
        
    if args.model == "diff_U_Net":
        model = UNet()
        model_name = "diff_U_Net"

    if args.model == "Resnet":
        if args.type =="S":
            hidden_dim = 32
            n_blocks =4
            model_name = "Resnet/4_blocks_32_hidden_dim"

        if args.type =="M":
            hidden_dim = 64
            n_blocks = 6
            model_name =  "Resnet/6_blocks_64_hidden_dim"

        if args.type =="L":
            hidden_dim = 96
            n_blocks = 9
            model_name =  "Resnet/9_blocks_96_hidden_dim"

        model = ResnetGenerator(input_nc=args.in_channels, output_nc=3, ngf=hidden_dim , n_blocks=n_blocks)
    # add aditional Loss to modelname
    if args.gaus_loss:
        model_name = model_name + '_gaus'

    if args.ssim_loss:
        model_name = model_name + '_ssim'

    if args.hist_loss:
        model_name = model_name + '_hist'

    # add Pix2Pix framwework to modelname
    if args.gan_framework:
        print('gan used')
        model_name = 'Pix2Pix/' + model_name

    if args.diff_model:
        if args.type =="S":
            args.diff_noise_steps = 500
        if args.type =="M":
            args.diff_noise_steps = 1000
        if args.type =="L":
            args.diff_noise_steps = 2000
        
        print('diffusionn model')
        model_name = 'diffusion_model/' + model_name


    return model , model_name





 
                
        
            
       
        

        
