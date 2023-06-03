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

    Clone the repository to your local machine.
    Navigate to the directory where the code is located.
    Run the script.

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
