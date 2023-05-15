import torch
from torch import nn
import numpy as np
import math

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
# MULTIHEAD SELF ATTENTION
#-----------------------------------------------------------------------------------------------
# 1) each patch [N ,P, embedding_dim] is mapped to 3 distinct vectors: q, k, and v (query, key, value)
# 2) for each patch, compute the dot product between its q vector with all of the k vectors, divide by the square root of the dimensionality of these vectors [sqrt(embedding_dim)]
#       ---> these are called attention cues
# 3) softmax(attention cues) --> sum(attention cues) = 1
# 4) multiply each attention cue with the v vectors associated with the different k vectors and sum all up
# 5) carry out num_head times 
#
# embedding_dim % num_head has to be 0 !!

class MSA(nn.Module):
    def __init__(self, d, num_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.num_heads = num_heads

        assert d % num_heads == 0, f"Can't divide dimension {d} into {num_heads} heads"

        d_head = int(d / num_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.num_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.num_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.num_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.num_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


#-----------------------------------------------------------------------------------------------
# VIT BLOCK
#-----------------------------------------------------------------------------------------------
class ViT_Block(nn.Module):
    def __init__(self, hidden_d, num_heads, mlp_ratio=4):
        super(ViT_Block, self).__init__()
        self.hidden_d = hidden_d
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, num_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
    

#-----------------------------------------------------------------------------------------------
# GENERATOR MODEL
#-----------------------------------------------------------------------------------------------

#
class Generator (nn.Module):
    def __init__(self, chw, patch_size, num_heads, num_blocks) -> None:
        super(Generator,self).__init__()
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

        assert chw[1] % patch_size[0] == 0, "Input shape not entirely divisible by patch_size"
        assert chw[2] % patch_size[1] == 0, "Input shape not entirely divisible by patch_size"

        # 1) Patch embedding
        self.patch_embedding = patch_embedding(in_channels=chw[0],
                                               patch_size=self.patch_size)
        
        # 2) Positional embedding
        self.pos_embedding = nn.Parameter(torch.tensor(get_positional_Embeddings((patch_size[0]**2)* chw[0] , self.embedding_dim)))
        self.pos_embedding.requires_grad = False

        # 3) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViT_Block(self.embedding_dim, self.num_heads) for _ in range(self.num_blocks)])

        self.unflatten = nn.Unflatten(2,(int(math.sqrt(self.embedding_dim)), int(math.sqrt(self.embedding_dim))))

        self.out = nn.Sequential(nn.Conv2d(self.chw[0], self.chw[0], 1, 1, 0),
                                  nn.Tanh()
                                )
        

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

        # unflatten patches 
        unflatten = self.unflatten(ViT_out)

        # change dimenstions to match input 
        out = unflatten.reshape([n,c,h,w])
        return out
    

#-----------------------------------------------------------------------------------------------
# DISCRIMINATOR
#-----------------------------------------------------------------------------------------------
# https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c   

class Discriminator(nn.Module):
    def __init__(self, chw, patch_size, num_heads, num_blocks, out_d=2):
        # Super constructor
        super(Discriminator, self).__init__()
# inputs :
# chw ------------> size of image [num_cahnnels, hight, width]
# patch_size -----> [w,h] : size of patch
# embedding_dim --> dimention of the linear projection
# num_heads ------> number of heads in the multihead self attention (MSA class)
# num_blocks -----> number of vision transformer blocks (ViT_Block class )
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        # maybe scale down with embedded_dim to hidden_dim
        self.embedding_dim = int((chw[1]/patch_size[0])**2)
        
        
        # Input and patches sizes
        assert chw[1] % patch_size[0] == 0, "Input shape not entirely divisible by patch_size"
        assert chw[2] % patch_size[1] == 0, "Input shape not entirely divisible by patch_size"
        
        
        # 1) Patch embedding
        self.patch_embedding = patch_embedding(in_channels=chw[0],
                                               patch_size=self.patch_size)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.embedding_dim))
        
        # 3) Positional embedding
        self.pos_embedding = nn.Parameter(torch.tensor(get_positional_Embeddings((patch_size[0]**2)* chw[0] +1 , self.embedding_dim)))

        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViT_Block(self.embedding_dim, self.num_heads) for _ in range(self.num_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
        )

    def forward(self, x):
        n, c, h, w = x.shape
        # input image x = [N, C, H, W]

        patches = self.patch_embedding(x)
        # P = patch_dim = (patch_size**2)* in_channels
        # D = lossles embedding_dim = (in_width/patch_size[0])**2
        # patches.shape = [N, P, D]

        
        tokens = torch.cat((self.class_token.expand(n, 1, -1), patches), dim=1)
        # Adding classification token to the tokens
        # tokens.shape = [N, P+1, D]


        # Adding positional embedding
        out = tokens + self.pos_embedding.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        out = out[:, 0]
       
        # apply mlp
        out = self.mlp(out)
        return out
    

def init_weights( module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    
    if isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    return module