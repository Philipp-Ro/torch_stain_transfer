import utils

model_architectures = {}
model_architectures["U-Net"] = ['my_UNet','my_UNet_color_loss_only']
model_architectures["diffusion_model"] = ['diff_1000t_loaded']
model_architectures["pix2pix"] = ['my_UNet_color_loss_only','Transformer']
model_architectures["ViT"] = ['Vit_1Block_4Mlp_4pat_mse+gaus']
model_architectures["swin_transfomer"] = ['2_stage_win4_96_dim_+gaus']
images = {}
images["img_num"]=[2,4,65,45]
images["patch_num"]= [0,0,0,0]
model, modelname = utils.get_publish_plot_img(images=images)
print(modelname)