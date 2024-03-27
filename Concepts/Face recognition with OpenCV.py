import cv2
import numpy as np

# Load a pre-trained model for face detection
face_detector = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')

# Load a pre-trained FaceNet model for feature extraction
# Assuming you have the FaceNet model loaded as `facenet_model`
facenet_model = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

def extract_face_features(image_path, face_detector, facenet_model):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Prepare the image for the face detector
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Loop over the face detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = image[startY:endY, startX:endX]
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            facenet_model.setInput(face_blob)
            vec = facenet_model.forward()

            # `vec` is the facial embedding (feature vector)
            return vec

# Example usage
image_path = 'path_to_your_image.jpg'
features = extract_face_features(image_path, face_detector, facenet_model)
print(features)


'''
from deepface import DeepFace
result = DeepFace.verify(img1_path = "path/to/image1.jpg", img2_path = "path/to/image2.jpg")
print("Are these images of the same person?: ", result["verified"])
'''