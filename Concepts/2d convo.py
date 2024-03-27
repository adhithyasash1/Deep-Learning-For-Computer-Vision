# Import necessary libraries
import numpy as np
import cv2
from PIL import Image

# Define functions as provided in the instructions
def standardize(image):
    eps = 1e-5
    return (image - np.mean(image)) / (np.std(image) + eps)

def linear_filter(image, filter_):
    image = np.array(image.convert('L'))
    image_height, image_width = image.shape[0], image.shape[1]
    filter_height, filter_width = filter_.shape[0], filter_.shape[1]
    result_height, result_width = (image_height - filter_height) + 1, (image_width - filter_width) + 1
    result = np.zeros((result_height, result_width))
    for i in range(result_height):
        for j in range(result_width):
            temp = standardize(image[i:filter_height+i, j:filter_width+j]) * filter_
            result[i][j] = np.sum(temp)
    return result

def convolution2D(image, kernel):
    kernel = kernel[::-1, ::-1]  # Flip the kernel for convolution operation
    result = linear_filter(image, kernel)
    return result

# Load the edges_image.png image for Sobel and Laplacian filters
edges_image_path = '/mnt/data/edges_image.png'
edges_image = Image.open(edges_image_path)

# Sobel Filter implementation
sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
response_v = convolution2D(edges_image, sobel_v)
response_h = convolution2D(edges_image, sobel_h)
G = np.abs(response_v) + np.abs(response_h)
min_index_sobel = np.unravel_index(np.argmin(G), G.shape)
max_index_sobel = np.unravel_index(np.argmax(G), G.shape)

# Laplacian Filter implementation
laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
response_L = convolution2D(edges_image, laplacian_filter)
min_index_laplacian = np.unravel_index(np.argmin(response_L), response_L.shape)
max_index_laplacian = np.unravel_index(np.argmax(response_L), response_L.shape)

# Load the corner_image.jpg image for Harris Corner Detection
corner_image_path = '/mnt/data/corner_image.jpg'
corner_image = cv2.imread(corner_image_path)
corner_image_rgb = cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
gray = cv2.cvtColor(corner_image, cv2.COLOR_BGR2GRAY)

# Harris Corner Detection implementation
block_size = 2
ksize = 3
k = 0.04
response_harris = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
dilated_response = cv2.dilate(response_harris, None)
threshold_harris = 0.1 * dilated_response.max()
num_corners = (dilated_response > threshold_harris).sum()

# Results
sobel_results = (min_index_sobel, max_index_sobel)
laplacian_results = (min_index_laplacian, max_index_laplacian)
harris_results = num_corners

(sobel_results, laplacian_results, harris_results)
