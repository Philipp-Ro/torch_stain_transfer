import torch.nn as nn
import numpy as np
#-----------------------------------------------------------------------------------------------
# RESIDUAL BLOCK
#-----------------------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
#-----------------------------------------------------------------------------------------------
# GENERATOR MODEL
#-----------------------------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, in_channels,hidden_dim, U_net_filter_groth,U_net_step_num, num_residual_blocks):
        super(Generator, self).__init__()

        
        # Inital Convolution:  3 * [img_height] * [img_width] ----> 64 * [img_height] * [img_width]
        #out_channels=64
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(in_channels), # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, hidden_dim, 2*in_channels+1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        channels = hidden_dim
        
        # Downsampling   64*256*256 -> 128*128*128 -> 256*64*64
        self.down = []
        for _ in range(U_net_step_num):
            out_channels = channels * U_net_filter_groth
            self.down += [
                nn.Conv2d(channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.down = nn.Sequential(*self.down)
        
        # Transformation (ResNet)  256*64*64
        self.trans = [ResidualBlock(channels) for _ in range(num_residual_blocks)]
        self.trans = nn.Sequential(*self.trans)
        
        # Upsampling  256*64*64 -> 128*128*128 -> 64*256*256
        self.up = []
        for _ in range(U_net_step_num):
            out_channels = channels // U_net_filter_groth
            self.up += [
                nn.Upsample(scale_factor=2), # bilinear interpolation
                nn.Conv2d(channels, out_channels, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.up = nn.Sequential(*self.up)
        
        # Out layer  64*256*256 -> 3*256*256
        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(channels, in_channels, 2*in_channels+1),
            nn.Tanh()

        )
    
    def forward(self, x):
       
        x = self.conv(x)
        x = self.down(x)
        x = self.trans(x)
        x = self.up(x)
        x = self.out(x)
        
        return x
    
#-----------------------------------------------------------------------------------------------
# DISCRIMINATOR
#-----------------------------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, disc_step_num, disc_filter_groth ,in_channels,hidden_dim):
        super(Discriminator, self).__init__()

        # Inital Convolution:  3 * [img_height] * [img_width] ----> 64 * [img_height] * [img_width]
        #out_channels=64
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(in_channels), # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, hidden_dim, 2*in_channels+1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        channels = hidden_dim
        
        # Downsampling   64*256*256 -> 128*128*128 -> 256*64*64
        self.down = []
        for _ in range(disc_step_num):
            out_channels = channels * disc_filter_groth
            self.down += [
                nn.Conv2d(channels, out_channels, kernel_size = 4, stride=2, padding=0),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.down = nn.Sequential(*self.down)

        self.conv_2 = nn.Conv2d(channels, 1, kernel_size = 2, stride=2, padding=0)
        
        self.mlp = nn.Linear(49,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        x = self.conv_2(x)
        x = x.view(x.shape[0],-1)
        out = self.mlp(x)
        #out = self.sigmoid(x)
        return out