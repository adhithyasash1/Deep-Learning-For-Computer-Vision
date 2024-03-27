pip install tensorflow

import numpy as np

# Generate a simple sequence of numbers
def generate_sequence(start, end):
    return np.arange(start, end)

# Prepare data for the LSTM
def prepare_data(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Define sequence
sequence = generate_sequence(1, 100)
# Choose a number of time steps
n_steps = 3
# Split into samples
X, y = prepare_data(sequence, n_steps)
# Reshape from [samples, timesteps] to [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Fit model
model.fit(X, y, epochs=200, verbose=1)

# Demonstrate prediction
x_input = np.array([97, 98, 99])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(f"Input Sequence: [97, 98, 99]\nPredicted: {yhat[0][0]}")
