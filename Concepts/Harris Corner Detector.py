import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image from file
image_path = 'path/to/your/image.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Harris Corner Detection parameters
block_size = 2
ksize = 3
k = 0.04

# Apply Harris Corner Detection
corner_response = cv2.cornerHarris(original_image, blockSize=block_size, ksize=ksize, k=k)

# Dilate the corner points to make them more visible
corner_response = cv2.dilate(corner_response, None)

# Threshold for detecting corners
threshold = 0.01 * corner_response.max()
corner_image = original_image.copy()
corner_image[corner_response > threshold] = 255

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(corner_image, cmap='gray')
plt.title('Harris Corner Detection')

plt.show()
