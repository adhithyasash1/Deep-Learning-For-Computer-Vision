import cv2
import numpy as np

# Corner Detection with Harris Algorithm

def detect_corners(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_img = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, marking the corners on the image
    image[dst > 0.01 * dst.max()] = [0, 255, 0]
    return image


# Laplacian Filter for Edge Detection

def apply_laplacian_filter(image):
    laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    response = convolution2D(image, laplacian_filter)
    return response