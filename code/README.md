# Immunohistochemical Image Generation using Deep Learning
This work is based on the data of the [BCI image generation challange](https://bci.grand-challenge.org/). It contains the code to train and evaluate multiple different network architectures for the task of translating HE stained images into IHC stained images of breast tissue while preserving the IHC score.

# Training
The training is started my executing the main.py file and setting the following arguments:
### --model
- U_Net
- ViT
- Swin
- Diffusion
- Resnet
### --type
- S
- M
- L
### --gan_freamework
- pix2pix
- score_gan
