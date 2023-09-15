# Emotion-Detection-CNN

This project aims to classify images as happy or sad using a convolutional neural network (CNN) built with TensorFlow. The model is designed to take in images of individuals and predict the emotion they are expressing.

## Prerequisites

To run this code, you will need the following installed:

    Python 3
    TensorFlow
    OpenCV
    Matplotlib

## Usage

The code can be run in any Python environment (like Jupyter Notebook or Google Colab). After setting up the environment and downloading the dataset, execute the code cell by cell.

It will train a CNN model on the provided dataset and save the trained model as 'emotiondetectionmodel.h5' in the 'models' directory. You can use this saved model for prediction on new images.

To predict the emotion of a new image, load the saved model using from tensorflow.keras.models import load_model. Then, preprocess the image using OpenCV and resize it to 256x256 pixels using tf.image.resize. Finally, feed the preprocessed image to the loaded model's predict function.

The script will perform the following steps:

    Load and preprocess the dataset of images from the specified directory.
    Split the dataset into training, validation, and test sets.
    Build the CNN model with three convolutional layers, max pooling, and two dense layers.
    Train the model on the training set, using the validation set for early stopping.
    Evaluate the model on the test set and print out accuracy, precision, and recall.
    Use the trained model to predict the emotions in any new image.

## Model Architecture

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


## Training

To train the CNN model, run the emotion_detection_cnn.ipynb notebook. The notebook uses TensorFlow's image_dataset_from_directory function to load the dataset and preprocess the images. The model architecture consists of three convolutional layers followed by a flattening layer, two dense layers, and an output layer with a sigmoid activation function. The model was trained using the Adam optimizer and binary cross-entropy loss function.
After training, the model's performance was evaluated on a test set using metrics such as precision, recall, and accuracy.

The model achieved high recall and accuracy rates, approximately 1.0. You can see the graphs of loss and accuracy during training in the following figures:

### Loss Graph:

![image](https://github.com/Nargesmohammadi/Emotion-Detection-CNN/assets/96385230/0395bd0b-1fcb-4f65-8bea-b883d79e9831)

### Accuracy Graph:

![image](https://github.com/Nargesmohammadi/Emotion-Detection-CNN/assets/96385230/9adb49b7-a1cf-4c9c-ac7c-c3e540f71512)


## Prediction

To use the trained model to make predictions on new images, you can load the model from the saved emodetectionmodel.h5 file and pass the image through the model. An example of predicting the emotion of a single image is provided in the notebook.

## Results

The trained model achieved an accuracy of 100% on the test set. An example of a correctly predicted image is shown below:

![image](https://github.com/Nargesmohammadi/Emotion-Detection-CNN/assets/96385230/fa357f83-54e5-4f93-958f-3e79f57facd6)

![image](https://github.com/Nargesmohammadi/Emotion-Detection-CNN/assets/96385230/345afde1-5d42-4d8e-bd56-b3990c456aba)


## Conclusion

In conclusion, this project demonstrates the use of CNNs for emotion detection in images. With further tuning and refinement, the model can potentially be used for real-world applications such as mood analysis in customer service or mental health settings.
