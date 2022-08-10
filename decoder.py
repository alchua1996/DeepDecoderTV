import torch.nn as nn
import torch
import torch.nn.functional as F
from tv_opt_layers.layers.general_tv_2d_layer import GeneralTV2DLayer

class ConvBlock(nn.Module):
    '''
    Class to create conv blocks:
    
    ReflectionPad2d(1) ---> Conv2d ---> affine bn ----> relu.
    
    Padding is set to 0 since relfection padding is added.
    Stride is set to 1 like w/ UNet/VGG since those results seem best.
    '''
    def __init__(self, in_channels, out_channels, activation):
        super(ConvBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size = 3, 
                              padding = 0, 
                              stride = 1)
        self.bn = nn.BatchNorm2d(out_channels,affine=True)
        self.activation = activation
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)
        

class DeepDecoder(nn.Module):
    def __init__(self, channels = [64]*5, activations = ['relu'] *4):
        '''
        channels: List with all the channels that the model will have.
        The paper uses all the same, it might be better to expand up though.
        
        Example of channels. A [64,128,64,64] will instantiate one conv 64 -> 128
        conv. Then a 124->54 conv. Then a 64 ->64 conv. Finally, the output layer 
        will be added.
        
        In the list of activations, here are the following options:
        'relu' adds a relu
        'leaky_relu' adds a leaky relu
        'tv_smooth' adds a tv_smooth function from the library
        'tv_sharp' adds a tv_sharp from the library
        Any other choice will default to relu. More options can be added later.
        '''      
        super(DeepDecoder, self).__init__()
        #build a model here by adding stuff to a sequential
        self.model = nn.Sequential()
        for i in range(len(channels)-1):
            #start by adding conv block
            if(activations[i] == 'relu'):
                self.model.add_module('conv_block_{}'.format(i+1),
                           ConvBlock(channels[i],channels[i+1], nn.ReLU()))
            elif(activations[i] == 'leaky_relu'):
                self.model.add_module('conv_block_{}'.format(i+1),
                           ConvBlock(channels[i],channels[i+1], nn.LeakyReLU()))
            elif(activations[i] == 'tv_smooth'):
                self.model.add_module('conv_block_{}'.format(i+1),
                          ConvBlock(channels[i],channels[i+1], GeneralTV2DLayer(lmbd_init=1)))
            elif(activations[i] == 'tv_smooth'):
                self.model.add_module('conv_block_{}'.format(i+1), 
                           ConvBlock(channels[i],channels[i+1], GeneralTV2DLayer(lmbd_init=1, mode = 'sharp')))
            else:
                self.model.add_module('conv_block_{}'.format(i+1), ConvBlock(channels[i],channels[i+1], nn.ReLU()))
            #add an upsample block
            self.model.add_module('upsample_{}'.format(i+1), nn.Upsample(scale_factor = 2, mode = 'bilinear'))
        #final conv layer
        self.model.add_module('final_pad', nn.ReflectionPad2d(1))
        self.model.add_module('final_conv', nn.Conv2d(channels[-1], 3, kernel_size = 3, padding = 0, stride = 1))
        self.model.add_module('sigmoid', nn.Sigmoid())   
  
    def forward(self, x):
        x = self.model(x)
        return x