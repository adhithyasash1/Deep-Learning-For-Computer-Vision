import numpy as np
import matplotlib.pyplot as plt
import scikit_image as ski

def gabor_filter(frequency, theta, sigma_x, sigma_y, gamma):
  """
  Creates a Gabor filter kernel.

  Args:
    frequency: Frequency of the filter.
    theta: Orientation of the filter (radians).
    sigma_x: Standard deviation in the x-direction.
    sigma_y: Standard deviation in the y-direction.
    gamma: Aspect ratio of the filter.

  Returns:
    A Gabor filter kernel as a 2D NumPy array.
  """
  x, y = np.meshgrid(np.arange(-sigma_x, sigma_x + 1, 0.1), np.arange(-sigma_y, sigma_y + 1, 0.1))
  X = x * np.cos(theta) + y * np.sin(theta)
  Y = -x * np.sin(theta) + y * np.cos(theta)
  gauss = np.exp(-((X**2 / (sigma_x**2)) + (Y**2 / (sigma_y**2))))
  return np.cos(2*np.pi*frequency*X) * gauss

# Set parameters
frequency = 0.2
theta = np.pi / 4
sigma_x = 2
sigma_y = 1
gamma = 0.5

# Generate filter
gabor_kernel = gabor_filter(frequency, theta, sigma_x, sigma_y, gamma)

# Visualize real and imaginary parts
plt.subplot(121)
plt.imshow(gabor_kernel.real, cmap='gray')
plt.title('Real Part')
plt.subplot(122)
plt.imshow(gabor_kernel.imag, cmap='gray')
plt.title('Imaginary Part')
plt.show()

# Visualize applying to an image (assuming you have an image loaded as 'img')
filtered_img = ski.filters.convolve(img, gabor_kernel)
plt.imshow(filtered_img, cmap='gray')
plt.show()

'''
The gabor_filter function creates a Gabor filter kernel based on input parameters:
frequency: Controls the wavelength of the sinusoidal pattern.
theta: Orientation of the filter.
sigma_x, sigma_y: Standard deviations for the Gaussian envelope.
gamma: Aspect ratio of the filter.
The returned kernel is a 2D array representing the real and imaginary parts of the complex-valued filter.
Visualizations show the individual real and imaginary parts, illustrating the sinusoidal pattern modulated by the Gaussian envelope.
Optionally, applying the filter to an image demonstrates its effect on extracting specific features based on its frequency and orientation.
'''