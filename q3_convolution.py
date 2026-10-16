"""
Question 3: Convolution Operations with Different Parameters
CS5720 - Home Assignment 2
 
Performs convolution on a 5x5 input matrix with a 3x3 kernel,
using all four combinations of stride (1, 2) and padding (VALID, SAME).
"""
 
import numpy as np
import tensorflow as tf
 
# 1. Define the 5x5 input matrix
input_matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)
 
# 2. Define the 3x3 kernel (Laplacian-style edge-detection kernel)
kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)
 
# Reshape for TensorFlow's conv2d: input needs shape (batch, height, width, channels)
# kernel needs shape (height, width, in_channels, out_channels)
input_tensor = input_matrix.reshape(1, 5, 5, 1)
kernel_tensor = kernel.reshape(3, 3, 1, 1)
 
def run_convolution(stride, padding):
    result = tf.nn.conv2d(
        input_tensor,
        kernel_tensor,
        strides=[1, stride, stride, 1],
        padding=padding
    )
    return result.numpy().reshape(result.shape[1], result.shape[2])
 
# 3. Perform convolution with all four combinations
print("Input Matrix (5x5):")
print(input_matrix)
print("\nKernel (3x3):")
print(kernel)
 
print("\n" + "="*50)
print("Stride = 1, Padding = 'VALID'")
print("="*50)
out1 = run_convolution(1, 'VALID')
print("Output shape:", out1.shape)
print(out1)
 
print("\n" + "="*50)
print("Stride = 1, Padding = 'SAME'")
print("="*50)
out2 = run_convolution(1, 'SAME')
print("Output shape:", out2.shape)
print(out2)
 
print("\n" + "="*50)
print("Stride = 2, Padding = 'VALID'")
print("="*50)
out3 = run_convolution(2, 'VALID')
print("Output shape:", out3.shape)
print(out3)
 
print("\n" + "="*50)
print("Stride = 2, Padding = 'SAME'")
print("="*50)
out4 = run_convolution(2, 'SAME')
print("Output shape:", out4.shape)
print(out4)
