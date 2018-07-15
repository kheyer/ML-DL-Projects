# Dog Breed CNN Visualizations

This project is about interpreting CNN results by visualizing multiclass probability predictions and layer activations using fast.ai. The files TSNE_distances.jpg and TSNE_probabilities.jpg show model predictions on 120 classes collapsed down to two dimensions to show class clustering.

The functions in cnn_viewer.py output content in the three folders in the repo:

  * composites_dog1 contains composite images for activations at a given layer of the network
  
  * individuals_dog1 contains examples of individual images from a given layer
  
  * animation_dog1 contains an mp4 animation of all images in all layers
  
cnn_viewer.py should work with any fast.ai convolutional model that works with three channel images. 

Some examples of activation images:

![Layer0](https://github.com/kheyer/ML-DL-Projects/blob/master/Dog%20Breed%20CNN%20Visualizations/individuals_dog1/layer0_image0.jpg)
![Layer1](https://github.com/kheyer/ML-DL-Projects/blob/master/Dog%20Breed%20CNN%20Visualizations/individuals_dog1/layer1_image4.jpg)
![Layer2](https://github.com/kheyer/ML-DL-Projects/blob/master/Dog%20Breed%20CNN%20Visualizations/individuals_dog1/layer2_image11.jpg)
![Layer3](https://github.com/kheyer/ML-DL-Projects/blob/master/Dog%20Breed%20CNN%20Visualizations/individuals_dog1/layer3_image11.jpg)
![Layer4](https://github.com/kheyer/ML-DL-Projects/blob/master/Dog%20Breed%20CNN%20Visualizations/individuals_dog1/layer4_image11.jpg)
![Layer5](https://github.com/kheyer/ML-DL-Projects/blob/master/Dog%20Breed%20CNN%20Visualizations/individuals_dog1/layer5_image1.jpg)
![Layer6](https://github.com/kheyer/ML-DL-Projects/blob/master/Dog%20Breed%20CNN%20Visualizations/individuals_dog1/layer6_image8.jpg)
![Layer7](https://github.com/kheyer/ML-DL-Projects/blob/master/Dog%20Breed%20CNN%20Visualizations/individuals_dog1/layer7_image6.jpg)
![Layer8](https://github.com/kheyer/ML-DL-Projects/blob/master/Dog%20Breed%20CNN%20Visualizations/individuals_dog1/layer8_image11.jpg)
