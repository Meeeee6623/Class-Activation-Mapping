# Class Activation Mapping

Class Activation Mapping (CAM) is a technique used in deep learning to visualize the features in an image that are most important for a specific prediction. In other words, it allows us to understand which parts of an image are most relevant for a particular classification. 

This project demonstrates CAM using the ResNet 50 model trained on the ImageNet dataset and transferred to the Fruits 360 dataset. The ResNet 50 model is a convolutional neural network (CNN) that has been pre-trained on a large dataset of images and can be used to classify images into one of 1000 different categories. In this project, the ResNet 50 model has been fine-tuned on the Fruits 360 dataset, which contains images of different types of fruit. By applying CAM to the ResNet 50 model, we can visualize the specific regions of the input image that are most important for each fruit classification.

To generate the Class Activation Maps, we first apply the ResNet 50 model to the input image to obtain the class prediction. Then, we compute the gradient of the class prediction with respect to the feature maps in the final convolutional layer of the model. Finally, we upsample the feature maps and apply a weighted sum to the input image, where the weights are determined by the gradients. The resulting Class Activation Map highlights the regions of the input image that are most important for the class prediction.

## Examples

Here are a few examples of Class Activation Maps generated by this project:

![Example 1](Fruits 360/cam_data/Cherry Wax Black.png)
- This is an example of a black cherry classification. The Class Activation Map shows that the network seems to focus almost entirely on the shine from the light onto the cherry, and not much on the rest of the fruit.

![Example 2](Fruits 360/cam_data/Corn.png)
- This is an example of a misclassified image, where the network has classified pieces of corn as a white onion and physalis with husk. The Class Activation Map helps us understand what the network focused on in order to make this incorrect classification, which can help us identify ways to improve the model's performance.

## References

- Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). Learning Deep Features for Discriminative Localization. In Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on (pp. 2921-2929). Institute of Electrical and Electronics Engineers Inc.
