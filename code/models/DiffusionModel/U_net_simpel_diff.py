# -------------------------------------------------------------------------------------------------------------------------
# Generator based on U-net 
# -------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import math
# ---------------------------------- double same convolution --------------------------------------------------------------
# -------> input_size of image = output_size of image
# -------> code was extended by batchnorm from the origilan paper
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
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
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], time_emb_dim=256
    ):
        super(U_net_Generator, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        self.relu  = nn.ReLU()

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x,t):
        skip_connections = []
        t = self.time_mlp(t)
        for down in self.downs:
            # --------------------------------------------------------------
            # create time embedding with correct dim 
            # --------------------------------------------------------------
            time_emb = self.relu(self.time_mlp(t))
            # Extend last 2 dimensions
            time_emb = time_emb[(..., ) + (None, ) * 2]
            # add time embedding 
            x = x+time_emb
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # --------------------------------------------------------------
            # create time embedding with correct dim 
            # --------------------------------------------------------------
            time_emb = self.relu(self.time_mlp(t))
            # Extend last 2 dimensions
            time_emb = time_emb[(..., ) + (None, ) * 2]
            # add time embedding 
            x = x+time_emb
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


def test():
    x = torch.randn((1, 3, 256, 256))
    model = U_net_Generator(in_channels=3, out_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()