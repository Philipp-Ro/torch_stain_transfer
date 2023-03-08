import torch 
from torch import nn
import math
import numpy as np

class PatchEmbedding(nn.Module):
####### patch embedding ###########
#   - in_channels are the input channels of the images for rgb = 3
#   - patch_size is the sizes of the patches for the embedding 
#   - img_size is the size of the input image (has to be quadratic)
#   - embedding_dim are the dimentions for the embedding with embedding_dim=0 it will be automaticly calculated 
    def __init__(self, in_channels, patch_size, embedding_dim):
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

# 1) the conv layer with kernel_size = stride writes out patches of the kernels the embedding_dim should be choosen as (patch_size**2)* in_channel so that no information is lost        
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=self.embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        
# 2) flatten the feature map into 1D
        self.flatten = nn.Flatten(start_dim=2, 
                                  end_dim=3)


    def forward(self, x):
      
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
# 3) permute the output tensor so that it has the form [batch_size, embedding_dim, num_patches] 
        return x_flattened
    
######### CREATE A MASK FOR POSITIONAL EMBEDDING ######################################  
def getPositionEmbedding(embedding_dim, num_patches, n=10000):
    # the n variable is scalling the values in the positional embedding in the attention is all you need paper n=10000 was choosen 
    p_embedding = torch.zeros((embedding_dim, num_patches))
    for k in range(embedding_dim):
        for i in torch.arange(int(num_patches/2)):
            denominator = np.power(n, 2*i/num_patches)
            p_embedding[k, 2*i] = np.sin(k/denominator)
            p_embedding[k, 2*i+1] = np.cos(k/denominator)
    
    return torch.unsqueeze(p_embedding, dim=0)
    

class Generator(nn.Module):
#---------------------------------- GENERATOR CLASS ---------------------------------------------------------------------------
#
#
# 1) set up the embedding of input img :
#        - img_size is the size of the input inmage of the generator 
#        - embedding_dim : the dimentions used also in the transformer
#        - patch_size : sizes of patches cut by the embedding [img_size % patch_size != 0]
#        - create positional embedding
#
# 2) set up the the transformer encoder layer class :
#        ---> https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer
#        - the d_model = embedding_dim so that the dimentions of the embedded image and the transformer network match
#        - nhead sets the number of heads for self attention in a transformer block 
#
# 3) set up encoder class:
#        --->  https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder
#        - use the encoder-layer set up in 2) 
#        - num_layers defines the number of encoder-layers in the encoder

    def __init__(self,img_size,embedding_dim, patch_size, in_channels, dropout_embedding, nhead,num_layers):
        super(Generator, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dropout_embedding = dropout_embedding
        self.nhead = nhead
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim


        #### testting the compatebility for img_size and patch_site 
        if img_size % patch_size == 0:
            self.patch_size = patch_size
        else : 
            print('img_size / patch_size has to have no rest')

        # number of patches in image for given patchsize 
        num_patches = (img_size * img_size) // patch_size**2 
        # number of valiables in input image ( num_channels* img_height* img_width)
        num_values = in_channels * img_size**2
        if num_values % num_patches == 0:
            self.embedding_dim =  int(num_values/num_patches)
        else:
            print('num_values / patch_num has to have no rest')

        # create patches from the imput image the output by the PatchEmbedding is : [batch_size, num_patches, embedding_dim ]
        # where as the embedding_dim is choosen as patch_size**2 * in_channels 
        self.patch_embedding = PatchEmbedding(in_channels=self.in_channels,patch_size=self.patch_size, embedding_dim=self.embedding_dim)
    
        self.num_patches = (img_size * img_size) // patch_size**2 

        self.embedding_dropout = nn.Dropout(p=self.dropout_embedding)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model= self.embedding_dim,
                                                               nhead=self.nhead ,
                                                               dim_feedforward=2048,
                                                               dropout=0.1,
                                                               activation="gelu",
                                                               batch_first=True,
                                                               norm_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(
                                                    encoder_layer=self.transformer_encoder_layer,
                                                    num_layers=self.num_layers)
        
        self.upsample = nn.PixelShuffle(self.num_patches)
        
        #self.linear = nn.Sequential(nn.Conv2d(self.embedding_dim, 3, 1, 1, 0))

    def forward(self, x):

        # Tensor input image: [batch_size, in_channel, img_size, img_size]

        x = self.patch_embedding(x)
        # Tensor enbedding: [batch_size, embedding_dim, num_patches]

        pos_embedding = getPositionEmbedding(self.embedding_dim, self.num_patches, n=1000)
        pos_embedding = pos_embedding.cuda()
        # positional enbedding: [batch_size, embedding_dim, num_patches]

        x = pos_embedding + x
        
        x = self.embedding_dropout(x)

        x = x.permute(0, 2, 1)
        
        x = self.transformer_encoder(x)

        x = x.permute(0, 2, 1)
        print(x.shape)

        x = x.view(1, self.embedding_dim, int(math.sqrt(self.num_patches)), int(math.sqrt(self.num_patches)))

        x = self.upsample(x)

        
        return x
    