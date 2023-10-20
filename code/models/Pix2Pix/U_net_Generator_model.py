# -------------------------------------------------------------------------------------------------------------------------
# Generator based on U-net 
# -------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
# ---------------------------------- double same convolution --------------------------------------------------------------
# -------> input_size of image = output_size of image
# -------> code was extended by batchnorm from the origilan paper
class DoubleConv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# ---------------------------------- U_net implimentation -----------------------------------------------------------------
# in_channels = num of channels in the input image 
# out_channels = num of channels in the input image 
# len(features) = number of steps of the down and upward part of the U_net
# features =  number of filters in the single step 
# each step is linkt with a skip connection to the opposite step 
class U_net_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64, steps=7, bottleneck_len=1):
        super(U_net_Generator, self).__init__()
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        self.intial_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=3,padding=1),
                                         nn.InstanceNorm2d(features)           
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        step_features= []
        step_feature = features
        for idx in range(steps):
            step_features.append(step_feature)
            in_channels = step_feature
            step_feature = int(step_feature*2)
            self.encoders.append(DoubleConv_block(in_channels, step_feature))
        step_features.append(step_feature)
        # Bottleneck
        for btn in range(bottleneck_len):
            self.bottleneck.append(DoubleConv_block(step_features[-1],step_features[-1]))

        # Up part of UNET
        in_channels=step_features[-1]*2
        for up_feature in reversed(step_features):
            self.decoders.append(DoubleConv_block(in_channels, int(up_feature/2)))
            self.upconvs.append(nn.ConvTranspose2d(int(in_channels/2), up_feature, kernel_size=2, stride=2))
            in_channels = up_feature
           


        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.intial_conv(x)

        for down in self.encoders:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        for btn in self.bottleneck:
            x = btn(x)

        idx=0
        for sk in reversed(skip_connections):
            x = self.upconvs[idx](x)
            concat_x = torch.cat((sk, x), dim=1)
            x = self.decoders[idx](concat_x)
            idx =idx+1
        x = self.final_conv(x) 
        return torch.sigmoid(x)
    


def test():
    x = torch.randn((1, 3, 256, 256))
    model = U_net_Generator(in_channels=3, out_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()