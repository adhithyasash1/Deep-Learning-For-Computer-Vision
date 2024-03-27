import numpy as np

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Example neural network with 1 hidden layer
# Input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# Output dataset            
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)  # For reproducibility

# Initialize weights randomly with mean 0
input_layer_size = 3  # Number of features
hidden_layer_size = 4  # Number of nodes in the hidden layer
output_layer_size = 1  # Number of nodes in the output layer

weights_input_hidden = 2 * np.random.random((input_layer_size, hidden_layer_size)) - 1
weights_hidden_output = 2 * np.random.random((hidden_layer_size, output_layer_size)) - 1

# Learning rate
learning_rate = 0.1

# Training the neural network
for iter in range(10000):

    # Forward pass
    input_layer = X
    hidden_layer = sigmoid(np.dot(input_layer, weights_input_hidden))
    output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output))
    
    # Backpropagation
    output_layer_error = y - output_layer
    output_layer_delta = output_layer_error * sigmoid_derivative(output_layer)
    
    hidden_layer_error = output_layer_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer)
    
    # Updating weights
    weights_hidden_output += learning_rate * hidden_layer.T.dot(output_layer_delta)
    weights_input_hidden += learning_rate * input_layer.T.dot(hidden_layer_delta)

# Display the output
print("Output after training:")
print(output_layer)

