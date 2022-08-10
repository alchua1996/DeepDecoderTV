# About This Repository
This is an unoffical implementation of [Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks](https://arxiv.org/abs/1810.03982) for denoising image. To improve the model, we attempted to add an inductive bias (since TV is a well studied phenomenon in mathematics for image denoising) by using learned total variation (TV) layers taken from [Total Variation Optimization Layers for Computer Vision](https://arxiv.org/abs/2204.03643). 

# Install/Usage Instructions
To use this code, follow the instructions to install the total variation activation functions from [the git of the authors](https://github.com/raymondyeh07/tv_layers_for_cv). The **folder of code "tv_opt_layers" is needed from the authors' github to run this code properly. It is not included here. ** 

The command for running the code is

'''
python denoise.py -i "image.png" -o "result.png" -c "64,64,64,64,64,64" -a "relu,relu,relu,relu,relu" -e 1500
'''

Switch "image.png" to y desired image to denoise. The output will be "result.png." The commands -c are for the number of channels in each layer and -a is for the activation functions. For this code to run properly, the the number of comma separated elements in -the must be 1 more than the number in -a. See decoder.py for more details of how the network works and the possible activation function choices. Lastly, -e stands for the number of epochs. From previous experience, 1500 seems to be the best.

# Example Runs and Discussion of Results
Below is the results of the original deep decoder on the image of a butterfly:
![vanilla_deep_decoder](https://user-images.githubusercontent.com/70219522/183818757-fbd3b47c-791d-4b61-8775-5547e2b1cac0.png)
Here is a run with some TV layers added:
![deep_decoder_tv](https://user-images.githubusercontent.com/70219522/183818659-052f2fda-ba15-4125-a272-9c4f7441382f.png)
Although the PSNR is slightly lower, the image quality is much higher because there are less artifiacts in the butterfly wings. 

# Citations and Aknowledgements
Heckel, R., and P. Hand. "Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks." ICLR 2019. 2018.

Yeh, Raymond A., et al. "Total Variation Optimization Layers for Computer Vision." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
