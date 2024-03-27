import tensorflow as tf
import numpy as np
import cv2

# Assuming you have a pre-trained model
model = tf.keras.models.load_model('path_to_your_model')

# Specify the last convolutional layer
layer_name = 'your_last_conv_layer'

# Create a model that outputs both the last convolutional layer and the final predictions
grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])

def compute_grad_cam_plus_plus(img_array, model, layer_name, predicted_class):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]
    
    # First-order gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Second-order and third-order gradients
    with tf.GradientTape() as tape_higher:
        tape_higher.watch(conv_outputs)
        with tf.GradientTape() as tape_inner:
            tape_inner.watch(conv_outputs)
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, predicted_class]
        first_order_grads = tape_inner.gradient(loss, conv_outputs)
    second_order_grads = tape_higher.gradient(first_order_grads, conv_outputs)
    third_order_grads = tape_higher.gradient(second_order_grads, conv_outputs)
    
    # Compute weights
    global_sum = np.sum(conv_outputs, axis=(0, 1, 2))
    alpha_num = second_order_grads * first_order_grads
    alpha_denom = 2.0 * second_order_grads + third_order_grads * global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom
    
    weights = np.maximum(first_order_grads, 0.0) * alphas
    weights = np.sum(weights, axis=(0, 1))
    
    # Compute the Grad-CAM++
    grad_cam_plus_plus = np.dot(conv_outputs[0], weights)
    
    # ReLU and normalization
    grad_cam_plus_plus = np.maximum(grad_cam_plus_plus, 0)
    grad_cam_plus_plus = cv2.resize(grad_cam_plus_plus, (img_array.shape[2], img_array.shape[1]))
    grad_cam_plus_plus = grad_cam_plus_plus / np.max(grad_cam_plus_plus)
    
    return grad_cam_plus_plus

# Example usage
img_array = preprocess_input('path_to_your_image')  # Make sure to preprocess the input image
predicted_class = np.argmax(model.predict(img_array))
grad_cam_plus_plus = compute_grad_cam_plus_plus(img_array, model, layer_name, predicted_class)

# Visualization
heatmap = np.uint8(255 * grad_cam_plus_plus)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + original_img

cv2.imshow('Grad-CAM++', superimposed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
