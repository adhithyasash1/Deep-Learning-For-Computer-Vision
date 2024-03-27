import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model('path_to_your_model')

# Specify the layer you want to visualize. Typically the last convolutional layer
layer_name = 'last_conv_layer_name'

# Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

# Get the gradient of the top predicted class for our input image with respect to the activations of the last conv layer
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(input_image)
    loss = predictions[:, np.argmax(predictions[0])]

# This is the gradient of the output neuron (top predicted or chosen) with respect to the output feature map of the last conv layer
output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]

# Compute the guided gradients
gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = gate_f * gate_r * grads

# The weights of the Grad-CAM
weights = tf.reduce_mean(guided_grads, axis=(0, 1))

# Compute the Grad-CAM
cam = np.ones(output.shape[0: 2], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * output[:, :, i]

# Post-processing: ReLU and normalization
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (input_image.shape[1], input_image.shape[2]))
cam = cam / np.max(cam)

# Visualize the heatmap
heatmap = np.uint8(255 * cam)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Superimpose the heatmap on original image
superimposed_img = heatmap * 0.4 + original_img

# Show the image
cv2.imshow('Grad-CAM', superimposed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
