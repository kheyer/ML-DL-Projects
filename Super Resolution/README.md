# Super Resolution

This notebook looks at using a U-Net model for super resolution and the impact of different loss functions and architecture choices. 
The loss functions investigated include [Perceptual loss](https://arxiv.org/pdf/1603.08155.pdf), 
[Gram Matrix loss](https://arxiv.org/pdf/1508.06576.pdf), and [L2-Wasserstein loss](https://github.com/VinceMarron/style_transfer). 
Architecture design looks at the impact of adding a [self attention](https://arxiv.org/abs/1805.08318) layer to the model.

![](https://github.com/kheyer/ML-DL-Projects/blob/master/Super%20Resolution/images/eye_composite_highdpi.png)
