# Skin Cancer Classification using CNN

## Introduction

This project aims to develop a skin cancer classification model using Convolutional Neural Networks (CNNs) to accurately identify various skin cancer diseases. The dataset used in this project consists of 2357 images of malignant and benign oncological diseases, including actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, and vascular lesion. The dataset is obtained from The International Skin Imaging Collaboration (ISIC), and the images are sorted based on their ISIC classifications.


This is the dataset - https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic 

## Data Preprocessing and Augmentation

The data is preprocessed and augmented to enhance the model's ability to generalize. Image augmentation techniques such as rotation, shear, zoom, and horizontal flip are applied to increase the diversity of the training data. This ensures that the model learns robust features from different variations of the skin cancer images.

## CNN Architecture with Dense Connectivity

The skin cancer classification model is built using a CNN architecture with dense connectivity. Dense connectivity allows for efficient information flow between layers, enabling better feature reuse and alleviating the vanishing gradient problem. The model's architecture consists of multiple dense blocks, each containing convolutional layers with dense connections.

## Attention Mechanisms

Attention mechanisms are incorporated into the CNN architecture to focus on relevant regions of the input image during classification. This attention mechanism helps the model to concentrate on critical features in the image, enhancing its ability to distinguish between different skin cancer types.

## Learning Rate Scheduling

To optimize the model's performance, learning rate scheduling is employed. The learning rate is adjusted dynamically during training, allowing the model to converge faster and potentially reach a more optimal solution. A lower learning rate is used during later stages of training to fine-tune the model and achieve better generalization.

## Gradient Clipping

Gradient clipping is utilized to prevent the gradients from exploding during backpropagation. This technique ensures stable training by capping the gradients to a maximum value, thus avoiding potential numerical instability.

## Computational Tools and Packages

The skin cancer classification model is developed using Python and various deep learning frameworks. TensorFlow and Keras are used for building and training the CNN model. NumPy and OpenCV are employed for data preprocessing and image augmentation. The model is trained on a powerful GPU to accelerate training and achieve high accuracy.

## How it Works

1. The dataset is loaded and preprocessed to prepare the images for training.
2. The CNN model with dense connectivity and attention mechanisms is constructed.
3. The model is trained using the training dataset with learning rate scheduling and gradient clipping.
4. The model is evaluated using the testing dataset to assess its performance and accuracy.
5. The trained model is used to predict the skin cancer type for given input images.

## Conclusion

The skin cancer classification model developed in this project demonstrates the effective use of CNNs, dense connectivity, attention mechanisms, learning rate scheduling, and gradient clipping to accurately classify different types of skin cancer. The model's high accuracy and ability to interpret relevant features make it a valuable tool for assisting medical professionals in diagnosing skin cancer diseases. The combination of computational methods and the understanding of skin cancer biology allows for a robust and efficient skin cancer classification system.