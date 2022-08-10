import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import torch.nn.functional as F

from tv_opt_layers.layers.general_tv_2d_layer import GeneralTV2DLayer

from decoder import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#get arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
ap.add_argument("-c", "--channels", required = True,
  help="list of activations, separated by a comma; see decoder file for details")
ap.add_argument("-a", "--activations", required = True,
  help="list of activations, separated by a comma; see decoder file for details")
ap.add_argument("-e", "--epochs", required = True,
  help="training iterations (suggest 1500)")
args = vars(ap.parse_args())

img = plt.imread(args['image'])
H,W,_ = img.shape
channels = [int(item) for item in args['channels'].split(',')]
n_layer = len(channels)-1
activations = [item for item in args['activations'].split(',')]
epochs = int(args['epochs'])

noise_img = random_noise(img, mode='gaussian', mean = 0, var=0.1, clip = 'True')
img_tensor = torch.Tensor(noise_img).unsqueeze(0).permute((0,3,1,2)).to(device)
#initialize white noise
noise = (0.1) * torch.rand((1,channels[0],int(H/2**n_layer),int(W/2**n_layer))).to(device)

denoiser = DeepDecoder(channels, activations).to(device)
optimizer = torch.optim.Adam(denoiser.parameters(),lr = 1e-2)

for epoch in range(epochs):
    optimizer.zero_grad()
    output = denoiser.forward(noise)
    loss = F.mse_loss(output, img_tensor) + total_variation_loss(output, weight = 1)

    if (epoch % 25 == 0):
      print('Epoch: {}', epoch)
      print('Loss: {}\n', loss.item())
    loss.backward()
    optimizer.step()
    #Regularization
    noise= noise + (1/(50))*torch.randn_like(noise) 

denoised_img = output.cpu().view(3,H,W).permute(1,2,0).detach().numpy()

plt.figure(figsize = (24,8))
plt.subplot(1,3,1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(1,3,2)
plt.imshow(noise_img)
plt.title('Noisy Image')
plt.subplot(1,3,3)
plt.imshow(denoised_img)
plt.title('Denoised Image (PSNR = {})'.format(np.round(compute_psnr(img, denoised_img),2)))
plt.savefig(args['output'])
