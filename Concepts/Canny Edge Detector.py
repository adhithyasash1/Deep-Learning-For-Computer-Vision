pip install opencv-python

import cv2
import matplotlib.pyplot as plt

# Load an image from file
image_path = 'path/to/your/image.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to the image to reduce noise and improve edge detection
blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)

# Apply Canny edge detector
edges = cv2.Canny(blurred_image, 50, 150)  # You can adjust the threshold values as needed

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')

plt.subplot(133)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')

plt.show()
