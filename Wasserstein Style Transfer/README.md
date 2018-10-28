# Style Transfer with a Wasserstein Metric 

This is a pytorch implementation of [Style Transfer as Optimum Transport](https://github.com/VinceMarron/style_transfer) by Vince Marron. 

This technique uses an L2-Wasserstein distance to optimize style transfer from one image to another. Similar to the [Gatys et. al](https://arxiv.org/abs/1508.06576) approach, the content image and the style image are passed through a pretrained convolutional model (in this cae VGG16) and the activations of both images at various layers are used as input to the loss function.

Unlike the Gatys approach, this technique does not use a combination of content loss and style loss. There is only one loss function - the L2-Wasserstein metric.

To simplify calculation of the metric, this technique assumes that the activations input into the loss function are multivariate Gaussian distributions modeled by the first two moments. This allows for a very straightforward calculation of the loss.

![wave](https://github.com/kheyer/ML-DL-Projects/blob/master/Wasserstein%20Style%20Transfer/styles/wave_comp.png)
![monet](https://github.com/kheyer/ML-DL-Projects/blob/master/Wasserstein%20Style%20Transfer/styles/monet_comp.png)
