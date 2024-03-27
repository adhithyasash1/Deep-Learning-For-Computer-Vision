import cv2
import numpy as np

img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

surf = cv2.xfeatures2d.SURF_create()

keypoints, descriptors = surf.detectAndCompute(img, None)

# If you have another image to match features with:
img2 = cv2.imread('second_image.jpg', cv2.IMREAD_GRAYSCALE)
keypoints2, descriptors2 = surf.detectAndCompute(img2, None)

# Match keypoints using Brute-Force matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors, descriptors2, k=2)

# Apply ratio test for robust matching
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])


'''
SURF is faster than SIFT while maintaining good robustness to scale, rotation, and illumination changes.
It's widely used for tasks similar to SIFT, including object recognition, image stitching, and scene matching.
Consider alternative feature detectors and descriptors based on your specific requirements and computational constraints.
Explore different matching techniques and parameter tuning for optimal results.
'''