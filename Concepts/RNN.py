import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate sequential data
def generate_sequence(start, end, step):
    return np.arange(start, end, step)

sequence = generate_sequence(0, 10, 0.1)

# Prepare the dataset
def prepare_data(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Choose a number of time steps
n_steps = 3
X, y = prepare_data(sequence, n_steps)

# Reshape from [samples, timesteps] to [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Define the RNN model
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X, y, epochs=200, verbose=0)

# Demonstrate prediction
x_input = np.array([0.7, 0.8, 0.9])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(f"Predicted: {yhat}")


'''from scratch pseudo'''

import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((output_size, 1))  # output bias

    def forward(self, inputs):
        """
        inputs - list of integers, where each integer is a one-hot encoded representation of the input sequence
        """
        h = np.zeros((self.Whh.shape[0], 1))  # initial hidden state
        outputs = []
        for i in inputs:
            x = np.zeros((self.Wxh.shape[1], 1))
            x[i] = 1  # Convert input to one-hot vector
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)  # hidden state update
            y = np.dot(self.Why, h) + self.by  # output vector
            outputs.append(y)
        return outputs, h

    def predict(self, inputs):
        outputs, _ = self.forward(inputs)
        return outputs[-1]  # return the last output

# Example usage
input_size = 10  # Suppose we're dealing with sequences of numbers 0-9
hidden_size = 100  # Size of the hidden layer
output_size = 10  # Output size is the same as the input size for prediction task

rnn = SimpleRNN(input_size, hidden_size, output_size)

# Example input sequence
inputs = [1, 2, 3, 4, 5]  # A simple sequence of numbers

# Forward pass
outputs = rnn.predict(inputs)
print(outputs)


def bptt(self, inputs, targets):
    # Perform forward pass and calculate the loss (omitted for brevity)
    # Initialize gradients as zero
    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    dhnext = np.zeros_like(self.bh)
    
    # Backpropagation through time
    for t in reversed(range(len(inputs))):
        # Calculate gradients (this is simplified; actual calculation depends on your loss function)
        # Here we need to calculate gradients for Wxh, Whh, Why, bh, by
        # Update these gradients based on the chain rule and propagate through time
        
        # Gradients for the output layer
        dy = np.copy(outputs[t])
        dy[targets[t]] -= 1  # assuming softmax and cross-entropy loss
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        
        # Gradients for the hidden layer (simplified, actual implementation will vary)
        dh = np.dot(self.Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(self.Whh.T, dhraw)
        
        # Add regularization terms (L2)
        dWxh += lambda * self.Wxh
        dWhh += lambda * self.Whh
        dWhy += lambda * self.Why
        
        # Gradient clipping
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
    
    # Update parameters using SGD
    for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
                             [dWxh, dWhh, dWhy, dbh, dby]):
        param -= learning_rate * dparam
