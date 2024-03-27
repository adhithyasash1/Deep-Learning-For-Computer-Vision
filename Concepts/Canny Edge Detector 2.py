import cv2
import numpy as np

img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(img, (5, 5), 0)

'''
This step reduces noise in the image, which can interfere with edge detection.
The kernel size (5x5) and standard deviation (0) can be adjusted based on your image and preferences.
'''

gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

'''
Sobel operator calculates horizontal (gx) and vertical (gy) gradients using a 3x3 kernel.
Gradients represent the direction and magnitude of intensity changes in the image.
'''

# Converts gradients to magnitude (strength of edge) and direction (angle of edge).
mag, dir = cv2.cartToPolar(gx, gy, angleInDegrees=True)

def nonmax_suppression(mag, dir):
    M, N = mag.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = dir * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # ... (check neighboring pixels based on angle for suppression)

    return Z

nms = nonmax_suppression(mag, dir)

'''
Suppresses pixels that are not local maxima in the gradient direction, thinning edges to single-pixel width.

The implementation details are omitted for brevity but involve checking neighboring pixels based on their angle and magnitude.
'''

low_thresh = 0.05 * mag.max()
high_thresh = 0.2 * mag.max()
canny = np.zeros_like(mag)
canny[(mag >= high_thresh) | ((mag >= low_thresh) & nms)] = 255

'''
Uses two thresholds:
High threshold identifies strong edges.
Low threshold identifies weaker edges that connect to strong edges.
Only pixels above the high threshold or pixels above the low threshold and connected to high-threshold pixels are kept as edges.
'''

cv2.imshow('Canny Edges', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
