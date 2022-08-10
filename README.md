# DeepDecoderTV
This is an unoffical implementation of [Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks](https://arxiv.org/abs/1810.03982) for denoising image. To improve the model, we attempted to add an inductive bias by using learned total variation (TV) layers taken from [Total Variation Optimization Layers for Computer Vision](https://arxiv.org/abs/2204.03643). 

To use this, follow the instructions to install the total variation activation functions from [the git of the authors](https://github.com/raymondyeh07/tv_layers_for_cv). 
The command for running the code is

'''
python denoise.py -i "image.png" -o "result.png" -c "64,64,64,64,64,64" -a "relu,relu,relu,relu,relu" -e 1500
'''

Switch "image.png" to your desired image to denoise. The output will be "result.png." The commands -c are for the number of channels in each layer and -a is for the activation functions. For this code to run properly, the the number of comma separated elements in -c must be 1 more than the number in -a. See decoder.py for more details of how the network works and the possible activation function choices.



