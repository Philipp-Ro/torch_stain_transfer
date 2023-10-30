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

# upconv block 
class up_conv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
# ---------------------------------- U_net implimentation -----------------------------------------------------------------
# in_channels = num of channels in the input image 
# out_channels = num of channels in the input image 
# len(features) = number of steps of the down and upward part of the U_net
# features =  number of filters in the single step 
# each step is linkt with a skip connection to the opposite step 
class U_net_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64, steps=7, attention=False):
        super(U_net_Generator, self).__init__()
        if steps < 3 or steps >=7:
            print('steps have to be between 3 and 6 ')
        else:
            self.steps = steps
            self.attention = attention
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        # outer step
        self.Conv1 = DoubleConv_block(in_channels=in_channels,out_channels=features)
        self.Conv_1x1 = nn.Conv2d(features,out_channels,kernel_size=1,stride=1,padding=0)

        # second step
        self.Conv2 = DoubleConv_block(in_channels=features,out_channels=int(features*2))
        self.Up2 = up_conv(in_channels=int(features*2), out_channels=features)
        if attention:
            self.Att2 = Attention_block(F_g=features,F_l=features,F_int=int(features/2))
        self.Up_conv2 = DoubleConv_block(in_channels=int(features*2), out_channels=features)

        # third step
        self.Conv3 = DoubleConv_block(in_channels=int(features*2),out_channels=int(features*4))
        self.Up3 = up_conv(in_channels=int(features*4),out_channels=int(features*2))
        if attention:
            self.Att3 = Attention_block(F_g=int(features*2),F_l=int(features*2),F_int=int(features))
        self.Up_conv3 = DoubleConv_block(in_channels=int(features*4),out_channels=int(features*2))

        # fourth step
        if self.steps >= 4:
            self.Conv4 = DoubleConv_block(in_channels=int(features*4),out_channels=int(features*8))
            self.Up4 = up_conv(in_channels=int(features*8),out_channels=int(features*4))
            if attention:
                self.Att4 = Attention_block(F_g=int(features*4),F_l=int(features*4),F_int=int(features*2))
            self.Up_conv4 = DoubleConv_block(in_channels=int(features*8),out_channels=int(features*4))

        # fith step 
        if self.steps >=5:
            self.Conv5 = DoubleConv_block(in_channels=int(features*8),out_channels=int(features*16))
            self.Up5 = up_conv(in_channels=int(features*16),out_channels=int(features*8))
            if attention:
                self.Att5 = Attention_block(F_g=int(features*8),F_l=int(features*8),F_int=int(features*4))
            self.Up_conv5 = DoubleConv_block(in_channels=int(features*16),out_channels=int(features*8))

        # sixth step
        if self.steps >=6:
            self.Conv6 = DoubleConv_block(in_channels=int(features*16),out_channels=int(features*32))
            self.Up6 = up_conv(in_channels=int(features*32),out_channels=int(features*16))
            if attention:
                self.Att6 = Attention_block(F_g=int(features*16),F_l=int(features*16),F_int=int(features*8))
            self.Up_conv6 = DoubleConv_block(in_channels=int(features*32),out_channels=int(features*16))


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        

        if self.steps>=4:
            x4 = self.Maxpool(x3)
            x4 = self.Conv4(x4)
            

        if self.steps>=5:
            x5 = self.Maxpool(x4)
            x5 = self.Conv5(x5)
            

        if self.steps==6:
            x6 = self.Maxpool(x5)
            x6 = self.Conv6(x6)

            d6 = self.Up6(x6)
            if self.attention:
                x5 = self.Att6(g=d6,x=x5)
            d6 = torch.cat((x5,d6),dim=1)        
            d6 = self.Up_conv6(d6)
            

        if self.steps>=5:
            if self.steps == 5: 
                in5 = x5
            else :
                in5 = d6

            d5 = self.Up5(in5)
            if self.attention:
                x4 = self.Att5(g=d5,x=x4)
            d5 = torch.cat((x4,d5),dim=1)        
            d5 = self.Up_conv5(d5)

        if self.steps>=4:
            if self.steps == 4: 
                in4 = x4
            else :
                in4 = d5
                
            d4 = self.Up4(in4)
            if self.attention:
                x3 = self.Att4(g=d4,x=x3)
            d4 = torch.cat((x3,d4),dim=1)
            d4 = self.Up_conv4(d4)


        if self.steps == 3: 
            in3 = x3
        else :
            in3 = d4
        d3 = self.Up3(in3)
        if self.attention:
            x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.attention:
            x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return torch.sigmoid(d1)

           
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    


def test():
    x = torch.randn((1, 3, 256, 256))
    model = U_net_Generator( in_channels=3, out_channels=3, features=32, steps=5, attention=True)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()