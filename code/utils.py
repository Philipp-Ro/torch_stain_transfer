
from torchvision import transforms
import torch 
import kornia
from architectures.U_net_Generator_model import U_net_Generator
from architectures.ViT_model import ViT_Generator
from architectures.SwinTransformer_model import SwinTransformer
from architectures.Unet_diff import UNet
from architectures.Resnet_gen import ResnetGenerator
import os
from pathlib import Path
import pickle

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

def build_model(args):
    if args.model == "None":
        model = "None"
        model_arch = "None"
        model_specs = "None"


    if args.model == "U_Net":
        attention = False
        if "S" in args.type:
            features= 16
            steps = 3
            model_arch ='U_Net/'
            model_specs='3step_16f'
            if "+att" in args.type:
                model_specs =model_specs+'+att'
                attention = True

        if "M" in args.type:
            features= 32
            steps = 4
            model_arch ='U_Net/'
            model_specs='4step_32f'
            
            if "+att" in args.type:
                model_specs =model_specs+'+att'
                attention = True

        if "L" in args.type:
            features= 64
            steps = 5
            model_arch ='U_Net/'
            model_specs='5step_64f'
            if "+att" in args.type:
                model_specs =model_specs+'+att'
                attention = True     

        if "X" in args.type:
            features= 128
            steps = 3
            model_arch ='U_Net/'
            model_specs='3step_128f'
            if "+att" in args.type:
                model_specs =model_specs+'+att'
                attention = True  

        model = U_net_Generator( in_channels=args.in_channels , out_channels=3, features=features, steps=steps, attention=attention)

    if args.model == "ViT":
        if args.type =="S":
            num_blocks= 1
            num_heads = 2
            model_arch ='ViT/'
            model_specs='1_block_2head'
          
        if args.type =="M":
            num_blocks =2
            num_heads = 4
            model_arch ='ViT/'
            model_specs='2_block_4head'

        if args.type =="L":
            num_blocks =3
            num_heads = 8
            model_arch ='ViT/'
            model_specs='3_block_8head'
        
        if args.img_resize:
            img_size_in = 256
        else:
            img_size_in = args.img_size

        model = ViT_Generator(  chw = [args.in_channels, img_size_in, img_size_in],
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
            model_arch ='Swin_T/'
            model_specs='2_stages_32_hidden_dim'

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
        
    if args.model == 'Diffusion':
        model = UNet()
        if args.type =="S":
            args.diff_noise_steps = 500
            model_arch ='diffusion_model/'
            model_specs='diff_U_Net_S'

        if args.type =="M":
            args.diff_noise_steps = 1000
            model_arch ='diffusion_model/'
            model_specs='diff_U_Net_M'

    if args.model == "Resnet":
        if args.type =="S":
            hidden_dim = 32
            n_blocks =4
            model_arch ='Resnet/'
            model_specs='4_blocks_32_hidden_dim'

        if args.type =="M":
            hidden_dim = 64
            n_blocks = 6
            model_arch ='Resnet/'
            model_specs='6_blocks_64_hidden_dim'

        if args.type =="L":
            hidden_dim = 96
            n_blocks = 9
            model_arch ="Resnet/"
            model_specs="9_blocks_96_hidden_dim"

        model = ResnetGenerator(input_nc=args.in_channels, output_nc=3, ngf=hidden_dim , n_blocks=n_blocks)
    # add aditional Loss to modelname
    if args.gaus_loss:
        model_specs = model_specs + '_gaus'

    if args.ssim_loss:
        model_specs = model_specs+ '_ssim'

    if args.hist_loss:
        model_specs = model_specs + '_hist'


    if args.gan_framework == 'pix2pix':
        model_framework = 'Pix2Pix/'

    if args.gan_framework == 'score_gan':
        model_framework = 'score_gan/'

    if args.gan_framework == 'None':
        model_framework = ''


    return model ,model_framework, model_arch, model_specs

def load_model_weights(args, model, model_name):
    if os.path.isdir(args.train_path):
        # load existing model 
        best_model_weights = os.path.join(args.train_path,'final_weights_gen.pth')
        model.load_state_dict(torch.load(best_model_weights))
        print(' ---------------------------------------------- ')
        print('pretrained ' +model_name+'  weights loaded')
    else:
        print(' ---------------------------------------------- ')
        print('no pretrained model found')

    return model

def set_paths(args  ,model_framework, model_arch, model_specs):
    result_dir = os.path.join(Path.cwd(),"masterthesis_results")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    # add gan_framwework to modelname
    if args.gan_framework != 'None':
        model_dir = os.path.join(result_dir,model_framework)
        model_name = model_framework + model_arch + model_specs

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
    else : 
        model_dir = result_dir
        model_name =  model_arch + model_specs

    full_model_dir = os.path.join(model_dir, model_arch)
    if not os.path.isdir(full_model_dir):
            os.mkdir(full_model_dir)

    args.train_path = os.path.join(model_dir,(model_arch+model_specs))
    args.train_eval_path = os.path.join(args.train_path,'train_plot_eval')
    args.test_eval_path = os.path.join(args.train_path,'test_plot_eval')
    args.c_path = os.path.join(args.train_path,"checkpoints")
    args.tp_path = os.path.join(args.train_path,'train_plots')
    

    if os.path.isdir(args.train_path):
        train_eval_path = os.path.join(args.train_path,'train_plot_eval')
        with open(train_eval_path, "rb") as fp:   
                train_plot_eval = pickle.load(fp)

        test_eval_path = os.path.join(args.train_path,'test_plot_eval')
        # load previous val_eval
        with open(test_eval_path, "rb") as fp:   
                test_plot_eval = pickle.load(fp)

    else:
        os.mkdir(args.train_path)
        os.mkdir(args.c_path)
        os.mkdir(args.tp_path)

        train_plot_eval =  {}
        train_plot_eval['MSE'] = []
        train_plot_eval['SSIM'] = []
        train_plot_eval['PSNR'] = []
        train_plot_eval['x'] = []

        test_plot_eval =  {}
        test_plot_eval['MSE'] = []
        test_plot_eval['SSIM'] = []
        test_plot_eval['PSNR'] = []
        test_plot_eval['x'] = []

    return args, model_name, train_plot_eval, test_plot_eval



def init_eval():
        eval = {}
        eval['total'] = {}
        eval['total']['MSE_mean'] = []
        eval['total']['SSIM_mean'] = []
        eval['total']['PSNR_mean'] = []
        eval['total']['num_img'] = 0

        eval['group_0']= {}
        eval['group_0']['MSE_mean'] = []
        eval['group_0']['SSIM_mean'] = []
        eval['group_0']['PSNR_mean'] = []
        eval['group_0']['num_img'] = 0

        eval['group_1']= {}
        eval['group_1']['MSE_mean'] = []
        eval['group_1']['SSIM_mean'] = []
        eval['group_1']['PSNR_mean'] = []
        eval['group_1']['num_img'] = 0

        eval['group_2']= {}
        eval['group_2']['MSE_mean'] = []
        eval['group_2']['SSIM_mean'] = []
        eval['group_2']['PSNR_mean'] = []
        eval['group_2']['num_img'] = 0

        eval['group_3']= {}
        eval['group_3']['MSE_mean'] = []
        eval['group_3']['SSIM_mean'] = []
        eval['group_3']['PSNR_mean'] = []
        eval['group_3']['num_img'] = 0

        eval['prediction_time']= 0
        eval['train_time'] = 0

        return eval


def init_epoch_eval_list():
    mse_list ={}
    ssim_list = {}
    psnr_list = {}

    mse_list['group_0'] = []
    mse_list['group_1'] = []
    mse_list['group_2'] = []
    mse_list['group_3'] = []
    mse_list['total'] = []

    ssim_list['group_0']= []
    ssim_list['group_1']= []
    ssim_list['group_2']= []
    ssim_list['group_3']= []
    ssim_list['total']= []

    psnr_list['group_0']= []
    psnr_list['group_1']= []
    psnr_list['group_2']= []
    psnr_list['group_3']= []
    psnr_list['total']= []

    return mse_list, ssim_list, psnr_list
           
def append_score_to_group_list(metric_score,eval_list, img_name):     
    if img_name[0].endswith("0.png"):
        eval_list['group_0'].append(metric_score)

    if img_name[0].endswith("1+.png"):
        eval_list['group_1'].append(metric_score)

    if img_name[0].endswith("2+.png"):
        eval_list['group_2'].append(metric_score)

    if img_name[0].endswith("3+.png"):
        eval_list['group_3'].append(metric_score)
    
    return eval_list

def write_result_in_file(resultfile_path, result, result_name):
    # write file
    with open(resultfile_path, 'w') as f: 
        f.write('-------------------------------------------------------' )
        f.write('\n Full test results on %s dataset: \n' % (result_name)) 
        f.write('\n trained for %s hours \n' % (result['train_time']/3600)) 
        f.write('\n average sampling time is  %s seconds \n' % (result['prediction_time'])) 
        f.write('-------------------------------------------------------\n' )
        for key_group,value_1 in result.items():
            if key_group ==  'prediction_time' or key_group == 'train_time':
                continue
            f.write('----------------------------------------------' )
            f.write('\n %s \n' % (key_group))
            for key, value in result[key_group].items(): 
                write_value = round(value,4)
                f.write('%s :  %s\n' % (key,write_value ))
            f.write('----------------------------------------------' ) 
            f.write('\n \n \n' ) 

        # close file
        f.close()

def get_IHC_score(img_name):
    if img_name[0].endswith("0.png"):
        score = torch.tensor(0)

    if img_name[0].endswith("1+.png"):
        score = torch.tensor(1)

    if img_name[0].endswith("2+.png"):
        score = torch.tensor(2)

    if img_name[0].endswith("3+.png"):
        score =torch.tensor(3)

    return score




        