#-----------------------------------------------------------------------------------------------
# Vision transformer Implementation 
#-----------------------------------------------------------------------------------------------
# Transformer block and multi head attention was adapted by the pytorch implementation from 
# https://pytorch.org/vision/main/models/vision_transformer.html
# which was inturn based on the paper ' An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'


import torch
from torch import nn
import numpy as np
import math
import torchvision
from einops import rearrange, reduce, repeat

#-----------------------------------------------------------------------------------------------
# MLP_Block
#-----------------------------------------------------------------------------------------------
class MLPBlock(torchvision.ops.misc.MLP):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)



#-----------------------------------------------------------------------------------------------
# PATCH EMBEDDING
#-----------------------------------------------------------------------------------------------
class patch_embedding(nn.Module):
    def __init__(self, in_channels, patch_size):
        super().__init__()
# ---------- inputs -------------------
# [N, in_channels, in_height, in_width]
# in_channels ------> int: number of channels in 
# patch_size -------> [w,h] : size of patch
#
# ---------- output -------------------
# P = patch_dim = (patch_size**2)* in_channels
# D = lossles embedding_dim = (in_width/patch_size[0])**2
# [N, P, D]
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_dim = (patch_size[0]**2)* in_channels

        assert patch_size[0] == patch_size[1], "patch_embedding is only for square patches"

        self.patcher = nn.Conv2d(   in_channels=in_channels,
                                    out_channels = self.patch_dim,
                                    kernel_size = patch_size[0],
                                    stride = patch_size[0],
                                    padding = 0)
        
        self.flatten = nn.Flatten(  start_dim=2, 
                                    end_dim=3)
        
    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 

        return x_flattened
    
#-----------------------------------------------------------------------------------------------
# POSITIONAL EMBEDDING
#-----------------------------------------------------------------------------------------------
def get_positional_Embeddings(sequence_length, embedding_dim):
    # the n variable is scalling the values in the positional embedding in the attention is all you need paper n=10000 was choosen 
    n = 10000
    result = torch.ones(sequence_length, embedding_dim)
    for i in range(sequence_length):
        for j in range(embedding_dim):
            result[i][j] = np.sin(i / (n ** (j / embedding_dim))) if j % 2 == 0 else np.cos(i / (n ** ((j - 1) / embedding_dim)))
    return result

#-----------------------------------------------------------------------------------------------
# ViT BLOCK
#-----------------------------------------------------------------------------------------------
class ViT_Block(nn.Module):
    def __init__(   self,
                    hidden_d, 
                    num_heads, 
                    attention_dropout, 
                    dropout, 
                    mlp_ratio,
    ):
        
        super(ViT_Block, self).__init__()
        self.hidden_d = hidden_d
        self.num_heads = num_heads

        # Attention block 
        self.norm1 = nn.LayerNorm(hidden_d)
        self.MHSA = torch.nn.MultiheadAttention(hidden_d , num_heads,dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP Block 
        mlp_dim = mlp_ratio * hidden_d
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = MLPBlock(hidden_d, mlp_dim, dropout)



        #self.norm2 = nn.LayerNorm(hidden_d)
        #self.mlp = nn.Sequential(
        #    nn.Linear(hidden_d, mlp_ratio * hidden_d),
        #    nn.GELU(),
        #    nn.Linear(mlp_ratio * hidden_d, hidden_d)
        #)

       
    def forward(self, input):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.norm1(input)
        x, _ = self.MHSA(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.norm2(x)
        y = self.mlp(y)
        return x + y


#-----------------------------------------------------------------------------------------------
# GENERATOR MODEL
#-----------------------------------------------------------------------------------------------
#
class ViT_Generator (nn.Module):
    def __init__(self, chw, patch_size, num_heads, num_blocks, attention_dropout, dropout, mlp_ratio) -> None:
        super(ViT_Generator,self).__init__()
# inputs :
# chw ------------> size of image [num_cahnnels, hight, width]
# patch_size -----> [w,h] : size of patch
# embedding_dim --> dimention of the linear projection
# num_heads ------> number of heads in the multihead self attention (MSA class)
# num_blocks -----> number of vision transformer blocks (ViT_Block class )


        # Attributes
        self.chw = chw # (C, H, W)
        self.patch_size = patch_size
        self.embedding_dim = int((chw[1]/patch_size[0])**2)
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        assert chw[1] % patch_size[0] == 0, "Input shape not entirely divisible by patch_size"
        assert chw[2] % patch_size[1] == 0, "Input shape not entirely divisible by patch_size"

        # 1) Patch embedding
        self.patch_embedding = patch_embedding(in_channels=chw[0],
                                               patch_size=self.patch_size)
        
        # 2) Positional embedding
        self.pos_embedding = nn.Parameter(torch.tensor(get_positional_Embeddings((patch_size[0]**2)* chw[0] , self.embedding_dim)))
        self.pos_embedding.requires_grad = False

        # 3) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViT_Block(self.embedding_dim, self.num_heads, self.attention_dropout, self.dropout, self.mlp_ratio) for _ in range(self.num_blocks)])

       
        self.transposed_conv = nn.ConvTranspose2d((patch_size[0]**2)* chw[0], self.chw[0], kernel_size=patch_size[0], stride=patch_size[0], padding=0)
        self.fianal_conv_layer = nn.Conv2d(chw[0],chw[0], kernel_size=1)

        
        


                                
        

    def forward(self, x):
        n, c, h, w = x.shape
        # input image x = [N, C, H, W]

        patches = self.patch_embedding(x)
        # P = patch_dim = (patch_size**2)* in_channels
        # D = lossles embedding_dim = (in_width/patch_size[0])**2
        # patches.shape = [N, P, D]

        pos_embed = self.pos_embedding.repeat(n, 1, 1)
        # calculating position embedding map
        # tokens have size [N,  P, D], we have to repeat the [P, D] positional encoding matrix N times
        # pos_embed = [N, P, D]

        ViT_in = patches + pos_embed
        # adding possition embedding 
       
        # Transformer Blocks
        for block in self.blocks:
            ViT_out = block(ViT_in)
        
        unsqueesed_out = rearrange(ViT_out, 'b p (d e) -> b p d e' , d=int(np.sqrt(self.embedding_dim)))

        out = self.transposed_conv(unsqueesed_out)
    
        out = self.fianal_conv_layer(out)
        return torch.sigmoid(out)

def test():
    x = torch.randn((1, 3, 256, 256)).cuda()
    model = ViT_Generator(chw=[3,256,256], patch_size=[32,32],num_heads=4,num_blocks=4 , attention_dropout = 0.2, dropout= 0.2, mlp_ratio=4).cuda()
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()