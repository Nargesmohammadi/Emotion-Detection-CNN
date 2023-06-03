# Emotion-Detection-CNN
Emotion Detection CNN

This project aims to classify images as happy or sad using a convolutional neural network (CNN) built with TensorFlow. The model is designed to take in images of individuals and predict the emotion they are expressing.
Prerequisites

To run this code, you will need the following installed:

    Python 3
    TensorFlow
    OpenCV
    Matplotlib

Usage
The code can be run in any Python environment (like Jupyter Notebook or Google Colab). After setting up the environment and downloading the dataset, execute the code cell by cell.

It will train a CNN model on the provided dataset and save the trained model as 'emotiondetectionmodel.h5' in the 'models' directory. You can use this saved model for prediction on new images.

To predict the emotion of a new image, load the saved model using from tensorflow.keras.models import load_model. Then, preprocess the image using OpenCV and resize it to 256x256 pixels using tf.image.resize. Finally, feed the preprocessed image to the loaded model's predict function.

The model achieved high recall and accuracy rates, approximately 1.0. You can see the graphs of loss and accuracy during training in the following figures:

Loss Graph:

![image](https://github.com/Nargesmohammadi/Emotion-Detection-CNN/assets/96385230/0395bd0b-1fcb-4f65-8bea-b883d79e9831)

Accuracy Graph:

![image](https://github.com/Nargesmohammadi/Emotion-Detection-CNN/assets/96385230/9adb49b7-a1cf-4c9c-ac7c-c3e540f71512)


The script will perform the following steps:

    Load and preprocess the dataset of images from the specified directory.
    Split the dataset into training, validation, and test sets.
    Build the CNN model with three convolutional layers, max pooling, and two dense layers.
    Train the model on the training set, using the validation set for early stopping.
    Evaluate the model on the test set and print out accuracy, precision, and recall.
    Use the trained model to predict the emotions in any new image.

Model Architecture

The CNN model has the following architecture:

    Input layer: 255x256x3
    Conv2D layer: 16 filters, 3x3 kernel size, stride=1, activation='relu'
    MaxPooling2D layer: pool size=2x2
    Conv2D layer: 32 filters, 3x3 kernel size, stride=1, activation='relu'
    MaxPooling2D layer: pool size=2x2
    Conv2D layer: 16 filters, 3x3 kernel size, stride=1, activation='relu'
    MaxPooling2D layer: pool size=2x2
    Flatten layer
    Dense layer: 256 nodes, activation='relu'
    Dense layer: 1 node, activation='sigmoid'
