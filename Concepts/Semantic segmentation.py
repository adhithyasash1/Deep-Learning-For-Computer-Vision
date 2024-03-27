import tensorflow as tf
import numpy as np
import cv2

# Load a pre-trained DeepLab model
model = tf.saved_model.load('path/to/deeplab_model')

# Function to run inference
@tf.function
def run_model(input_tensor):
    return model.signatures['serving_default'](input_tensor)

# Load and preprocess an image
image_path = 'path/to/your/image.jpg'
img = cv2.imread(image_path)
img_resized = tf.image.resize(img, [513, 513])
input_tensor = tf.convert_to_tensor(img_resized, dtype=tf.float32)[tf.newaxis, ...]

# Run the model
results = run_model(input_tensor)

# Post-process results to generate segmentation mask
segmentation_mask = tf.argmax(results['SemanticPredictions'], axis=-1)
segmentation_mask = segmentation_mask[0, :, :, tf.newaxis].numpy()

# Visualization of the segmentation mask
mask_colored = np.zeros_like(img)
for class_id in np.unique(segmentation_mask):
    mask_colored[segmentation_mask == class_id] = np.random.randint(0, 255, size=3)
cv2.imshow('Segmentation Mask', mask_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
