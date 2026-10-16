"""
Question 5, Task 1: Implement AlexNet Architecture
CS5720 - Home Assignment 2
 
Builds a simplified AlexNet model using TensorFlow/Keras and prints
its model summary.
"""
 
import tensorflow as tf
from tensorflow.keras import layers, models
 
def build_alexnet(input_shape=(227, 227, 3), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
 
        # Conv Layer 1
        layers.Conv2D(96, kernel_size=(11, 11), strides=4, activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),
 
        # Conv Layer 2
        layers.Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),
 
        # Conv Layers 3-5
        layers.Conv2D(384, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Conv2D(384, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),
 
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
 
        # Output layer
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model
 
alexnet = build_alexnet()
alexnet.summary()
