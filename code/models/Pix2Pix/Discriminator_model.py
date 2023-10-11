# code from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/Pix2Pix
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic block"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fx = self.conv(x)
        
        if self.isn is not None:
            fx = self.isn(fx)
            
        fx = self.lrelu(fx)
        return fx


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super().__init__()
        self.block1 = BasicBlock(in_channels*2, features, norm=False)
        self.block2 = BasicBlock(features, features*2)
        self.block3 = BasicBlock(features*2, features*4)
        self.block4 = BasicBlock(features*4, features*8)
        self.block5 = nn.Conv2d(features*8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        # blocks forward
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        
        return fx


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    input = torch.cat((x,y), 1)

    model = Discriminator(in_channels=3)
    preds = model(input)
    print(preds.shape)


if __name__ == "__main__":
    test()