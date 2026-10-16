"""
Question 4, Task 1: Edge Detection Using Sobel Filter
CS5720 - Home Assignment 2
 
Applies Sobel-X and Sobel-Y filters to a grayscale image for edge detection.
"""
 
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
# 1. Create/load a grayscale sample image
# (Using a generated synthetic image with clear shapes/edges so Sobel
#  filtering has something meaningful to detect. Replace this block with
#  cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE) to use a real photo.)
image = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(image, (40, 40), (160, 160), 255, -1)
cv2.circle(image, (100, 100), 40, 120, -1)
 
# 2. Define Sobel filters (as given in the assignment)
sobel_x_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)
 
sobel_y_kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)
 
# Apply Sobel filters using cv2.filter2D (manual kernel application),
# matching the exact kernels given in the assignment.
sobel_x = cv2.filter2D(image, cv2.CV_64F, sobel_x_kernel)
sobel_y = cv2.filter2D(image, cv2.CV_64F, sobel_y_kernel)
 
# Convert back to displayable 8-bit format
sobel_x_display = cv2.convertScaleAbs(sobel_x)
sobel_y_display = cv2.convertScaleAbs(sobel_y)
 
# 3. Display original image and filtered images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")
 
axes[1].imshow(sobel_x_display, cmap="gray")
axes[1].set_title("Sobel-X (Vertical Edges)")
axes[1].axis("off")
 
axes[2].imshow(sobel_y_display, cmap="gray")
axes[2].set_title("Sobel-Y (Horizontal Edges)")
axes[2].axis("off")
 
plt.tight_layout()
plt.savefig("sobel_edge_detection.png", dpi=150)
print("Saved sobel_edge_detection.png")
print("\nOriginal image shape:", image.shape)
print("Sobel-X output shape:", sobel_x.shape)
print("Sobel-Y output shape:", sobel_y.shape)
