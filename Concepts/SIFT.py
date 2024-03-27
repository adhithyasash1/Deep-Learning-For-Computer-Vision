import cv2
import numpy as np

img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(img, None)

'''
keypoints: Collection of keypoints, each with attributes like location (x, y), size, angle, etc.
descriptors: 128-dimensional feature vectors for each keypoint, describing surrounding image patch.
'''

# If you have another image to match features with:
img2 = cv2.imread('second_image.jpg', cv2.IMREAD_GRAYSCALE)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Match keypoints using Brute-Force matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors, descriptors2, k=2)

# Apply ratio test for robust matching
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

'''
SIFT is robust to scale, rotation, and illumination changes.
It's often used for object recognition, image stitching, and scene matching.
Consider alternative feature detectors and descriptors like SURF, ORB, or AKAZE based on your specific use case.
Explore different matching techniques and parameter tuning for optimal results.
'''