# Residual UNets

Adding a residual path to the decoder section of a UNet appears to slightly improve performance and segmentation quality when compared 
to a non-residual model for the same number of epochs.

Masks without residual connection:

![](https://github.com/kheyer/ML-DL-Projects/blob/master/Experiments/Residual%20UNets/media/without_residual.png)

Masks with residual connection:

![](https://github.com/kheyer/ML-DL-Projects/blob/master/Experiments/Residual%20UNets/media/with_residual.png)
