from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

# Step 1: Data Preparation
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Filter the dataset to keep only the images of '0' as normal
x_train_normal = x_train[y_train == 0]
x_test_normal = x_test[y_test == 0]
x_test_anomaly = x_test[y_test != 0]

# Step 2: Model Architecture
input_img = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(encoded)
x = Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()


autoencoder.fit(x_train_normal, x_train_normal,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_normal, x_test_normal))


# Predict the reconstruction of the test set
reconstructed = autoencoder.predict(x_test)

# Calculate the mean squared error of reconstruction
mse = np.mean(np.power(x_test - reconstructed, 2), axis=(1, 2, 3))

# Define a threshold for anomaly detection
threshold = np.quantile(mse, 0.99)  # for example, using the 99th percentile as threshold

# Detect anomalies
anomalies = mse > threshold


import matplotlib.pyplot as plt

# Example of plotting original and reconstructed images
n = 10  # number of digits to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
