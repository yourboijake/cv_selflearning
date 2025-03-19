import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MD_LSTM_Cell:
    def __init__(self, input_size, hidden_size):
        # Initialize weights and biases
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Initialize weights for input gate
        self.W_i = np.random.randn(input_size + 2 * hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((1, hidden_size))
        
        # Initialize weights for forget gate (one for each direction)
        self.W_f1 = np.random.randn(input_size + 2 * hidden_size, hidden_size) * 0.01
        self.W_f2 = np.random.randn(input_size + 2 * hidden_size, hidden_size) * 0.01
        self.b_f1 = np.zeros((1, hidden_size))
        self.b_f2 = np.zeros((1, hidden_size))
        
        # Initialize weights for output gate
        self.W_o = np.random.randn(input_size + 2 * hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((1, hidden_size))
        
        # Initialize weights for cell state
        self.W_c = np.random.randn(input_size + 2 * hidden_size, hidden_size) * 0.01
        self.b_c = np.zeros((1, hidden_size))
        
    def forward(self, x, h1_prev, h2_prev, c1_prev, c2_prev):
        # Concatenate input and previous hidden states
        combined = np.concatenate((x, h1_prev, h2_prev), axis=1)
        
        # Input gate
        i = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
        
        # Forget gates (one for each direction)
        f1 = self.sigmoid(np.dot(combined, self.W_f1) + self.b_f1)
        f2 = self.sigmoid(np.dot(combined, self.W_f2) + self.b_f2)
        
        # Output gate
        o = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
        
        # Cell candidate
        c_candidate = np.tanh(np.dot(combined, self.W_c) + self.b_c)
        
        # Cell state
        c = f1 * c1_prev + f2 * c2_prev + i * c_candidate
        
        # Hidden state
        h = o * np.tanh(c)
        
        return h, c
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def get_params(self):
        return [self.W_i, self.W_f1, self.W_f2, self.W_o, self.W_c, 
                self.b_i, self.b_f1, self.b_f2, self.b_o, self.b_c]
    
    def set_params(self, params):
        self.W_i, self.W_f1, self.W_f2, self.W_o, self.W_c, \
        self.b_i, self.b_f1, self.b_f2, self.b_o, self.b_c = params

class MD_LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize the 2D LSTM cells
        self.lstm_cell = MD_LSTM_Cell(input_size, hidden_size)
        
        # Initialize weights for output layer
        self.W_out = np.random.randn(hidden_size, output_size) * 0.01
        self.b_out = np.zeros((1, output_size))
        
    def forward(self, x):
        # x shape: (batch_size, height, width, channels)
        batch_size, height, width, channels = x.shape
        
        # Initialize hidden and cell states
        h_tl = np.zeros((batch_size, height, width, self.hidden_size))  # top-left
        c_tl = np.zeros((batch_size, height, width, self.hidden_size))
        h_tr = np.zeros((batch_size, height, width, self.hidden_size))  # top-right
        c_tr = np.zeros((batch_size, height, width, self.hidden_size))
        h_bl = np.zeros((batch_size, height, width, self.hidden_size))  # bottom-left
        c_bl = np.zeros((batch_size, height, width, self.hidden_size))
        h_br = np.zeros((batch_size, height, width, self.hidden_size))  # bottom-right
        c_br = np.zeros((batch_size, height, width, self.hidden_size))
        
        # Process in 4 directions
        # Top-Left to Bottom-Right
        for i in range(height):
            for j in range(width):
                h1_prev = np.zeros((batch_size, self.hidden_size)) if i == 0 else h_tl[:, i-1, j]
                h2_prev = np.zeros((batch_size, self.hidden_size)) if j == 0 else h_tl[:, i, j-1]
                c1_prev = np.zeros((batch_size, self.hidden_size)) if i == 0 else c_tl[:, i-1, j]
                c2_prev = np.zeros((batch_size, self.hidden_size)) if j == 0 else c_tl[:, i, j-1]
                
                h_tl[:, i, j], c_tl[:, i, j] = self.lstm_cell.forward(x[:, i, j], h1_prev, h2_prev, c1_prev, c2_prev)
        
        # Top-Right to Bottom-Left
        for i in range(height):
            for j in range(width-1, -1, -1):
                h1_prev = np.zeros((batch_size, self.hidden_size)) if i == 0 else h_tr[:, i-1, j]
                h2_prev = np.zeros((batch_size, self.hidden_size)) if j == width-1 else h_tr[:, i, j+1]
                c1_prev = np.zeros((batch_size, self.hidden_size)) if i == 0 else c_tr[:, i-1, j]
                c2_prev = np.zeros((batch_size, self.hidden_size)) if j == width-1 else c_tr[:, i, j+1]
                
                h_tr[:, i, j], c_tr[:, i, j] = self.lstm_cell.forward(x[:, i, j], h1_prev, h2_prev, c1_prev, c2_prev)
        
        # Bottom-Left to Top-Right
        for i in range(height-1, -1, -1):
            for j in range(width):
                h1_prev = np.zeros((batch_size, self.hidden_size)) if i == height-1 else h_bl[:, i+1, j]
                h2_prev = np.zeros((batch_size, self.hidden_size)) if j == 0 else h_bl[:, i, j-1]
                c1_prev = np.zeros((batch_size, self.hidden_size)) if i == height-1 else c_bl[:, i+1, j]
                c2_prev = np.zeros((batch_size, self.hidden_size)) if j == 0 else c_bl[:, i, j-1]
                
                h_bl[:, i, j], c_bl[:, i, j] = self.lstm_cell.forward(x[:, i, j], h1_prev, h2_prev, c1_prev, c2_prev)
        
        # Bottom-Right to Top-Left
        for i in range(height-1, -1, -1):
            for j in range(width-1, -1, -1):
                h1_prev = np.zeros((batch_size, self.hidden_size)) if i == height-1 else h_br[:, i+1, j]
                h2_prev = np.zeros((batch_size, self.hidden_size)) if j == width-1 else h_br[:, i, j+1]
                c1_prev = np.zeros((batch_size, self.hidden_size)) if i == height-1 else c_br[:, i+1, j]
                c2_prev = np.zeros((batch_size, self.hidden_size)) if j == width-1 else c_br[:, i, j+1]
                
                h_br[:, i, j], c_br[:, i, j] = self.lstm_cell.forward(x[:, i, j], h1_prev, h2_prev, c1_prev, c2_prev)
        
        # Combine the results from all directions
        h_combined = h_tl[:, -1, -1] + h_tr[:, -1, 0] + h_bl[:, 0, -1] + h_br[:, 0, 0]
        
        # Output layer
        output = np.dot(h_combined, self.W_out) + self.b_out
        
        return output
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        y_pred_softmax = self.softmax(y_pred)
        log_likelihood = -np.log(y_pred_softmax[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def train(self, X, y, epochs=10, learning_rate=0.01, batch_size=32):
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Create batches
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                loss = self.cross_entropy_loss(y_pred, y_batch)
                total_loss += loss
                
                # Very simple gradient descent update
                # In a real implementation, we would compute gradients properly
                # This is just a placeholder for demonstration
                # Normally we'd use backpropagation through time (BPTT)
                
                # Update weights with small random perturbations
                # (This is NOT a proper training method, just a demo)
                params = self.lstm_cell.get_params()
                new_params = []
                for param in params:
                    new_params.append(param - learning_rate * np.random.randn(*param.shape) * 0.01)
                self.lstm_cell.set_params(new_params)
                
                # Update output layer weights
                self.W_out -= learning_rate * np.random.randn(*self.W_out.shape) * 0.01
                self.b_out -= learning_rate * np.random.randn(*self.b_out.shape) * 0.01
            
            avg_loss = total_loss / (len(X) / batch_size)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses

# Example usage on a simple problem: MNIST digit recognition
def main():
    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Reshape data to 2D grid format (8x8 images)
    X = X.reshape(-1, 8, 8, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = MD_LSTM(input_size=1, hidden_size=32, output_size=10)
    losses = model.train(X_train, y_train, epochs=5, learning_rate=0.01, batch_size=16)
    
    # Plot the loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
    # Make predictions
    y_pred = model.forward(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_classes == y_test)
    print(f"Test accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
