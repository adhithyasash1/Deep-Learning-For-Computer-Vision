import cv2
import numpy as np

img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate gradients
# Using Sobel operator for gradients
Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Calculate Harris Matrix
Ix2 = Ix**2
Iy2 = Iy**2
Ixy = Ix * Iy

kernel = np.array([[1, 4, 1], [4, 16, 4], [1, 4, 1]])  # Example kernel

Sxx = cv2.filter2D(Ix2, -1, kernel)
Syy = cv2.filter2D(Iy2, -1, kernel)
Sxy = cv2.filter2D(Ixy, -1, kernel)

# Calculate Harris Corner Response
k = 0.04  # Harris detector parameter
det = (Sxx * Syy) - (Sxy**2)
trace = Sxx + Syy
R = det - k * (trace**2)

# Apply Thresholding and Find Corners
threshold = 0.01 * R.max()  # Adjust threshold as needed
corners = np.argwhere(R > threshold)

# Draw circles at corner locations
for corner in corners:
    x, y = corner
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)  # Red circles for corners

cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Experiment with different kernel sizes and thresholds for fine-tuning results.
Consider using adaptive thresholding techniques for more robust corner detection in varying image conditions.
The Harris detector is often used as a basis for further feature extraction and tracking algorithms.
'''
