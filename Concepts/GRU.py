import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class GRUCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.Wz = np.random.randn(hidden_dim, input_dim)
        self.Uz = np.random.randn(hidden_dim, hidden_dim)
        self.bz = np.zeros((hidden_dim, 1))
        
        self.Wr = np.random.randn(hidden_dim, input_dim)
        self.Ur = np.random.randn(hidden_dim, hidden_dim)
        self.br = np.zeros((hidden_dim, 1))
        
        self.Wh = np.random.randn(hidden_dim, input_dim)
        self.Uh = np.random.randn(hidden_dim, hidden_dim)
        self.bh = np.zeros((hidden_dim, 1))
        
    def forward(self, x, h_prev):
        # Update gate
        z = sigmoid(np.dot(self.Wz, x) + np.dot(self.Uz, h_prev) + self.bz)
        
        # Reset gate
        r = sigmoid(np.dot(self.Wr, x) + np.dot(self.Ur, h_prev) + self.br)
        
        # Candidate hidden state
        h_hat = tanh(np.dot(self.Wh, x) + np.dot(self.Uh, (r * h_prev)) + self.bh)
        
        # Final hidden state
        h = (1 - z) * h_prev + z * h_hat
        
        return h


class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input gate weights
        self.Wi = np.random.randn(hidden_dim, input_dim)
        self.Ui = np.random.randn(hidden_dim, hidden_dim)
        self.bi = np.zeros((hidden_dim, 1))
        
        # Forget gate weights
        self.Wf = np.random.randn(hidden_dim, input_dim)
        self.Uf = np.random.randn(hidden_dim, hidden_dim)
        self.bf = np.zeros((hidden_dim, 1))
        
        # Output gate weights
        self.Wo = np.random.randn(hidden_dim, input_dim)
        self.Uo = np.random.randn(hidden_dim, hidden_dim)
        self.bo = np.zeros((hidden_dim, 1))
        
        # Cell state weights
        self.Wc = np.random.randn(hidden_dim, input_dim)
        self.Uc = np.random.randn(hidden_dim, hidden_dim)
        self.bc = np.zeros((hidden_dim, 1))
        
    def forward(self, x, h_prev, c_prev):
        # Input gate
        i = sigmoid(np.dot(self.Wi, x) + np.dot(self.Ui, h_prev) + self.bi)
        
        # Forget gate
        f = sigmoid(np.dot(self.Wf, x) + np.dot(self.Uf, h_prev) + self.bf)
        
        # Output gate
        o = sigmoid(np.dot(self.Wo, x) + np.dot(self.Uo, h_prev) + self.bo)
        
        # Candidate memory cell
        c_hat = tanh(np.dot(self.Wc, x) + np.dot(self.Uc, h_prev) + self.bc)
        
        # Final memory cell
        c = f * c_prev + i * c_hat
        
        # Final hidden state
        h = o * tanh(c)
        
        return h, c
