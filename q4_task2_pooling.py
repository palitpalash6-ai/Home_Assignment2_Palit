"""
Question 4, Task 2: Max Pooling and Average Pooling
CS5720 - Home Assignment 2
 
Demonstrates 2x2 Max Pooling and Average Pooling on a random 4x4 matrix.
"""
 
import numpy as np
import tensorflow as tf
 
# 1. Create a random 4x4 matrix as an input "image"
np.random.seed(42)
input_matrix = np.random.randint(0, 10, size=(4, 4)).astype(np.float32)
 
# Reshape for TensorFlow pooling layers: (batch, height, width, channels)
input_tensor = input_matrix.reshape(1, 4, 4, 1)
 
# 2. Apply 2x2 Max Pooling
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
max_pooled = max_pool(input_tensor).numpy().reshape(2, 2)
 
# 3. Apply 2x2 Average Pooling
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
avg_pooled = avg_pool(input_tensor).numpy().reshape(2, 2)
 
# 4. Print original, max-pooled, and average-pooled matrices
print("Original 4x4 Matrix:")
print(input_matrix)
 
print("\nMax Pooled (2x2) Matrix:")
print(max_pooled)
 
print("\nAverage Pooled (2x2) Matrix:")
print(avg_pooled)
