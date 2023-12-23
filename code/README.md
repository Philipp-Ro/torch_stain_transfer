# Immunohistochemical Image Generation using Deep Learning
This work is based on the data of the [BCI image generation challange](https://bci.grand-challenge.org/). It contains the code to train and evaluate multiple different network architectures for the task of translating HE stained images into IHC stained images of breast tissue while preserving the IHC score.

# Training
The training is started my executing the main.py file and setting the following arguments:

## Network setup:
### --model
there are fvie architectures to choose from for the generator network:
- U_Net
- ViT
- Swin
- Diffusion
- Resnet
### --type
there are 3 types availeble for each architecture:
- S
- M
- L
### --gan_freamework
additionally a gan_framework can be used with an discriminator to inhance the generator performance:
- pix2pix
- score_gan

## Optimizer setup:
### --lr 
the learning rate can be adjusted the default is 3e-5
### --beta1
### --beta2
the beta variables for the adam optimizer can be adjusted the default are beta1= 0.5 and beta2= 0.999

## training loop and preprocessing setup : 
### --img_size 
the imagesize for the training the orininal img is 1024x1024 the default setting is 256 so that the oringinal images will be sliced into 256x256 patches and each patch is fed into the the network. The training loop will interate over one patch in all images per epoch and than switch the patch in the next epoch
### --img_resize
if a resizes of the original image is desired instead of the patching you can pass the img_resize flag default is False
### --in_channels
if in_channels is put on 1 the image will be in grayscale
### --img_transforms
the img_transfroms flag is a list strings which indicate the preprocessing for the original images which happens before the the resize or patchify of the image the default is ["colorjitter",'horizontal_flip','vertical_flip']
### --num_epochs 
the input for the number of epoch default 100
### --decay_epoch
the input for the decay epoch default 80
### --batch_size 
default 1
### --device 
the device where the training is running on the default is "cuda"


## loss adjustment for the training 
there are options to add additional losses to the total loss of the generator 
### --gaus_loss
this flag will add a gausian blurr loss default is False
### --ssim_loss
this flag will add a ssim loss which is calculated as (1-ssim) default is False
### --hist_loss
this flass will add a histogramm loss 

## dir setup for the training
put the paths for the data folders
### --train_data
directory to the train data with two folders in is as HE for He_imges and IHC for the IHC_imgs 
the images in the both folders have to have corresponding names ending with the IHC_score 
### --test_data
directory to the train data with two folders in is as HE for He_imges and IHC for the IHC_imgs 

## Testing setup
flags for testing
## --quant_eval_only 
this flag will initiate the quantitativ evaluation of the network 
## -qual_eval_only
this flag will initiate the qualitativ evaluation of the network 

