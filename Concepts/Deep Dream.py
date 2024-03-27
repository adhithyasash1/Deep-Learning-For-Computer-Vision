import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained model
model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# Choose the layer(s) to enhance. For example, a mix of layers can provide interesting effects
names = ['mixed3', 'mixed5']
layers = [model.get_layer(name).output for name in names]

# Create a model for feature extraction
dream_model = tf.keras.Model(inputs=model.input, outputs=layers)

def compute_loss(image, model):
    # Pass the image through the model to retrieve the activations
    layer_activations = model(image)
    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    return tf.reduce_sum(losses)

@tf.function
def deepdream_step(img, model, learning_rate):
    with tf.GradientTape() as tape:
        # 'watch' the input image
        tape.watch(img)
        # Compute loss
        loss = compute_loss(img, model)
    # Calculate the gradient of the loss with respect to the pixels of the input image.
    gradients = tape.gradient(loss, img)
    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8 
    # Update the image by adding the gradients (this enhances the dream)
    img += gradients * learning_rate
    return loss, img

def run_deep_dream_simple(img, model, steps=100, learning_rate=0.01):
    # Convert the image to a tensor
    img = tf.constant(np.array(img))
    img = tf.expand_dims(img, axis=0)
    img = tf.cast(img, dtype=tf.float32)
    for step in range(steps):
        loss, img = deepdream_step(img, model, learning_rate)
        if step % 10 == 0:
            print(f"Step {step}, Loss {loss}")
    plt.figure(figsize=(12, 12))
    plt.imshow(deprocess(img))
    plt.show()

# Function to preprocess and deprocess images
def deprocess(img):
    img = 255*(img - img.min()) / (img.max() - img.min())
    return tf.cast(img, tf.uint8)

# Example usage
# Load an image
img = plt.imread('your_image.jpg')
# Run DeepDream
run_deep_dream_simple(img, dream_model)
