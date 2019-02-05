# Texture Synthesis with a Wasserstein Metric

This is a texture synthesis application of a style transfer technique I demonstrated in a [previous notebook](https://github.com/kheyer/ML-DL-Projects/tree/master/Wasserstein%20Style%20Transfer).

For texture synthesis, a random image is generated. The random image is then optimized to minimize the L2-Wasserstein distance between 
the generated image and the target style image. The input features to the distance metric are perceptual features extracted 
using a pretrained VGG16 network. The L2-Wasserstein distance between the input and target features is calculated by assuming the 
extracted features are drawn from a multivariate Gaussian distribution.

![grid](https://github.com/kheyer/ML-DL-Projects/blob/master/Wasserstein%20Texture%20Synthesis/Media/grid.png)
