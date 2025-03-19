import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class CTCLoss:
    def __init__(self, blank_idx=0):
        """
        Initialize CTC loss.
        
        Args:
            blank_idx: Index of the blank symbol in the vocabulary
        """
        self.blank_idx = blank_idx
        
    def prepare_targets(self, targets, alphabet_size):
        """
        Prepare targets by adding blanks between each character and at the beginning/end.
        
        Args:
            targets: List of target label indices
            alphabet_size: Size of the alphabet including blank
            
        Returns:
            Extended targets with blanks
        """
        # Add blanks between each pair of labels and at the beginning/end
        extended_targets = [self.blank_idx]
        for char in targets:
            extended_targets.extend([char, self.blank_idx])
        return extended_targets
    
    def forward_algorithm(self, log_probs, targets, input_lengths, target_lengths):
        """
        Forward algorithm for CTC.
        
        Args:
            log_probs: Log probabilities from the network [batch, time, alphabet_size]
            targets: Target sequences
            input_lengths: Length of each input sequence
            target_lengths: Length of each target sequence
            
        Returns:
            Forward variables and total log probability
        """
        batch_size = log_probs.shape[0]
        max_time = log_probs.shape[1]
        
        batch_log_probs = []
        
        for b in range(batch_size):
            # Get the current sequence
            T = input_lengths[b]  # Length of this input
            log_prob_seq = log_probs[b, :T]  # [T, alphabet_size]
            target = targets[b, :target_lengths[b]]  # Actual target sequence
            
            # Prepare target with blanks
            alphabet_size = log_prob_seq.shape[1]
            extended_target = self.prepare_targets(target, alphabet_size)
            L = len(extended_target)
            
            # Initialize forward variables
            forward = np.ones((T, L)) * -np.inf
            
            # Initialization
            forward[0, 0] = log_prob_seq[0, self.blank_idx]
            if L > 1:
                forward[0, 1] = log_prob_seq[0, extended_target[1]]
            
            # Forward pass
            for t in range(1, T):
                for s in range(L):
                    # Case 1: same label or blank
                    forward[t, s] = forward[t-1, s]
                    
                    # Case 2: transition from previous label
                    if s > 0:
                        forward[t, s] = np.logaddexp(forward[t, s], forward[t-1, s-1])
                    
                    # Case 3: skip if repeated label (not blank)
                    if s > 1 and extended_target[s] != self.blank_idx and extended_target[s] != extended_target[s-2]:
                        forward[t, s] = np.logaddexp(forward[t, s], forward[t-1, s-2])
                    
                    # Add current log probability
                    forward[t, s] += log_prob_seq[t, extended_target[s]]
            
            # Total log probability is the sum of the last two positions
            if L > 1:
                log_p = np.logaddexp(forward[T-1, L-1], forward[T-1, L-2])
            else:
                log_p = forward[T-1, L-1]
                
            batch_log_probs.append(log_p)
        
        # Return negative mean log probability as loss
        return -np.mean(batch_log_probs)
    
    def backward_algorithm(self, log_probs, targets, input_lengths, target_lengths):
        """
        Backward algorithm for CTC.
        
        Args:
            log_probs: Log probabilities from the network [batch, time, alphabet_size]
            targets: Target sequences
            input_lengths: Length of each input sequence
            target_lengths: Length of each target sequence
            
        Returns:
            Backward variables
        """
        batch_size = log_probs.shape[0]
        max_time = log_probs.shape[1]
        alphabet_size = log_probs.shape[2]
        
        backward_vars = []
        
        for b in range(batch_size):
            # Get the current sequence
            T = input_lengths[b]  # Length of this input
            log_prob_seq = log_probs[b, :T]  # [T, alphabet_size]
            target = targets[b, :target_lengths[b]]  # Actual target sequence
            
            # Prepare target with blanks
            extended_target = self.prepare_targets(target, alphabet_size)
            L = len(extended_target)
            
            # Initialize backward variables
            backward = np.ones((T, L)) * -np.inf
            
            # Initialization (last time step)
            backward[T-1, L-1] = 0  # log(1) = 0
            if L > 1:
                backward[T-1, L-2] = 0  # Both final positions can end the sequence
            
            # Backward pass
            for t in range(T-2, -1, -1):
                for s in range(L):
                    # Case 1: same label or blank
                    backward[t, s] = backward[t+1, s] + log_prob_seq[t+1, extended_target[s]]
                    
                    # Case 2: transition to next label
                    if s < L-1:
                        backward[t, s] = np.logaddexp(backward[t, s], 
                                                    backward[t+1, s+1] + log_prob_seq[t+1, extended_target[s+1]])
                    
                    # Case 3: skip if possible (consecutive non-blanks)
                    if s < L-2 and extended_target[s+2] != self.blank_idx and extended_target[s+2] != extended_target[s]:
                        backward[t, s] = np.logaddexp(backward[t, s], 
                                                     backward[t+1, s+2] + log_prob_seq[t+1, extended_target[s+2]])
            
            backward_vars.append(backward)
        
        return backward_vars
    
    def decode_greedy(self, log_probs, input_lengths):
        """
        Greedy decoding - just take the highest probability label at each step.
        
        Args:
            log_probs: Log probabilities from the network [batch, time, alphabet_size]
            input_lengths: Length of each input sequence
            
        Returns:
            Decoded sequences
        """
        batch_size = log_probs.shape[0]
        max_time = log_probs.shape[1]
        
        decoded = []
        
        for b in range(batch_size):
            # Get the current sequence
            T = input_lengths[b]  # Length of this input
            log_prob_seq = log_probs[b, :T]  # [T, alphabet_size]
            
            # Greedy decoding (take most likely label at each step)
            best_path = np.argmax(log_prob_seq, axis=1)
            
            # Collapse repeated labels and remove blanks
            prev = -1
            collapsed = []
            for p in best_path:
                if p != prev:  # New label or first occurrence
                    if p != self.blank_idx:  # Not blank
                        collapsed.append(p)
                    prev = p
            
            decoded.append(collapsed)
        
        return decoded
    
    def decode_beam_search(self, log_probs, input_lengths, beam_size=10):
        """
        Beam search decoding - keep track of top-k paths.
        
        Args:
            log_probs: Log probabilities from the network [batch, time, alphabet_size]
            input_lengths: Length of each input sequence
            beam_size: Number of beams to maintain
            
        Returns:
            Decoded sequences
        """
        batch_size = log_probs.shape[0]
        decoded = []
        
        for b in range(batch_size):
            # Get the current sequence
            T = input_lengths[b]  # Length of this input
            log_prob_seq = log_probs[b, :T]  # [T, alphabet_size]
            
            # Initialize with empty sequence
            beam = [{'sequence': [], 'score': 0.0, 'last_char': self.blank_idx}]
            
            # Iterate through time steps
            for t in range(T):
                new_beam = []
                
                for candidate in beam:
                    seq, score, last_char = candidate['sequence'], candidate['score'], candidate['last_char']
                    
                    # Try each label
                    for c in range(log_prob_seq.shape[1]):
                        new_score = score + log_prob_seq[t, c]
                        
                        if c == self.blank_idx:  # Blank
                            new_seq = seq.copy()
                            new_last = self.blank_idx
                        elif c == last_char:  # Repeated character
                            new_seq = seq.copy()
                            new_last = c
                        else:  # New character
                            new_seq = seq + [c]
                            new_last = c
                        
                        new_beam.append({'sequence': new_seq, 'score': new_score, 'last_char': new_last})
                
                # Sort and keep top beam_size
                new_beam.sort(key=lambda x: x['score'], reverse=True)
                beam = new_beam[:beam_size]
            
            # Return the best sequence
            decoded.append(beam[0]['sequence'])
        
        return decoded

class HandwritingRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Simple RNN for handwriting recognition.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            output_size: Number of output classes (alphabet size)
        """
        # Initialize weights with small random values
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input -> Hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden -> Hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden -> Output
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def forward(self, inputs):
        """
        Forward pass through the RNN.
        
        Args:
            inputs: Input sequences [batch, time, features]
            
        Returns:
            Log probabilities for each time step and hidden states
        """
        batch_size, time_steps, input_size = inputs.shape
        
        # Preallocate memory for outputs and states
        h = np.zeros((batch_size, time_steps + 1, self.hidden_size))
        log_probs = np.zeros((batch_size, time_steps, self.output_size))
        
        # Process each sequence in the batch
        for b in range(batch_size):
            for t in range(time_steps):
                # Input at this time step
                x = inputs[b, t].reshape(-1, 1)
                
                # RNN step
                h[b, t+1] = np.tanh(self.Wxh @ x + self.Whh @ h[b, t].reshape(-1, 1) + self.bh).reshape(-1)
                
                # Output probabilities
                y = self.Why @ h[b, t+1].reshape(-1, 1) + self.by
                log_probs[b, t] = y.reshape(-1)
        
        # Apply log softmax
        log_probs = np.log(softmax(log_probs))
        
        return log_probs, h
    
    def train_step(self, inputs, targets, input_lengths, target_lengths, learning_rate=0.001):
        """
        Train the RNN using CTC loss.
        
        Args:
            inputs: Input sequences [batch, time, features]
            targets: Target sequences [batch, max_target_len]
            input_lengths: Length of each input sequence
            target_lengths: Length of each target sequence
            learning_rate: Learning rate for gradient updates
            
        Returns:
            Loss value
        """
        # Forward pass
        log_probs, h = self.forward(inputs)
        
        # Compute CTC loss
        ctc = CTCLoss()
        loss = ctc.forward_algorithm(log_probs, targets, input_lengths, target_lengths)
        
        # Backpropagation through time (simplified gradient calculation)
        # In a real implementation, you would use automatic differentiation
        
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # Calculate gradients (simplified for demonstration)
        # In practice, you would compute proper gradients from the CTC loss
        batch_size, time_steps, _ = inputs.shape
        
        for b in range(batch_size):
            # Simple gradient estimation - this is NOT accurate for CTC loss
            # Just for illustrative purposes
            dh_next = np.zeros((self.hidden_size, 1))
            
            for t in reversed(range(time_steps)):
                # Gradient of output layer
                dy = log_probs[b, t].reshape(-1, 1).copy()
                
                # Very rough approximation of CTC gradient
                # Increase probability of correct label, decrease others
                if t < target_lengths[b]:
                    correct_idx = targets[b, t]
                    dy[correct_idx] -= 1
                
                # Gradient of weights from hidden to output
                dWhy += dy @ h[b, t+1].reshape(1, -1)
                dby += dy
                
                # Gradient of hidden state
                dh = self.Why.T @ dy + dh_next
                
                # Backprop through tanh
                dh_raw = (1 - h[b, t+1]**2) * dh
                
                # Gradient of weights
                dWxh += dh_raw @ inputs[b, t].reshape(1, -1)
                dWhh += dh_raw @ h[b, t].reshape(1, -1)
                dbh += dh_raw
                
                # Save gradient for next iteration
                dh_next = self.Whh.T @ dh_raw
        
        # Update weights with simple SGD
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
        
        return loss

# Example usage with synthetic handwriting data
def generate_synthetic_data(batch_size=16, time_steps=50, feature_size=8, num_classes=27, max_label_len=10):
    """
    Generate synthetic handwriting recognition data.
    
    Args:
        batch_size: Number of sequences in batch
        time_steps: Maximum length of input sequences
        feature_size: Feature dimension of input
        num_classes: Number of output classes (alphabet size)
        max_label_len: Maximum length of output labels
    
    Returns:
        inputs, targets, input_lengths, target_lengths
    """
    # Generate random input features
    inputs = np.random.randn(batch_size, time_steps, feature_size)
    
    # Generate random target sequences (representing characters)
    # 0 is blank, 1-26 represent a-z
    target_lengths = np.random.randint(1, max_label_len + 1, size=batch_size)
    targets = np.zeros((batch_size, max_label_len), dtype=np.int32)
    
    for i in range(batch_size):
        # Generate random characters (1-26)
        targets[i, :target_lengths[i]] = np.random.randint(1, num_classes, size=target_lengths[i])
    
    # Ensure input sequences are longer than target sequences
    input_lengths = target_lengths * 5  # Roughly 5 frames per character
    input_lengths = np.minimum(input_lengths, time_steps)
    
    return inputs, targets, input_lengths, target_lengths

def train_handwriting_model():
    """Train a simple RNN model for handwriting recognition using CTC loss."""
    # Parameters
    input_size = 8  # Feature dimension
    hidden_size = 64  # RNN hidden state size
    output_size = 27  # a-z + blank
    num_epochs = 10
    batch_size = 16
    
    # Create model
    model = HandwritingRNN(input_size, hidden_size, output_size)
    ctc = CTCLoss()
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 5  # Number of batches per epoch
        
        for _ in range(num_batches):
            # Generate synthetic batch
            inputs, targets, input_lengths, target_lengths = generate_synthetic_data(
                batch_size=batch_size, time_steps=50, feature_size=input_size, 
                num_classes=output_size, max_label_len=10
            )
            
            # Forward and backward pass
            loss = model.train_step(inputs, targets, input_lengths, target_lengths)
            total_loss += loss
        
        # Average loss for this epoch
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate on a single example
        if epoch % 2 == 0:
            test_input, test_target, test_input_len, test_target_len = generate_synthetic_data(
                batch_size=1, time_steps=50, feature_size=input_size, 
                num_classes=output_size, max_label_len=10
            )
            
            log_probs, _ = model.forward(test_input)
            decoded = ctc.decode_greedy(log_probs, test_input_len)
            
            # Convert to characters for display
            alphabet = "-abcdefghijklmnopqrstuvwxyz"
            decoded_str = ''.join([alphabet[idx] for idx in decoded[0]])
            target_str = ''.join([alphabet[idx] for idx in test_target[0, :test_target_len[0]]])
            
            print(f"  Target: {target_str}")
            print(f"Decoded: {decoded_str}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('CTC Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    return model, losses

# Visualization function for handwriting recognition
def visualize_ctc_alignment(log_probs, decoded, input_length):
    """
    Visualize the CTC alignment between input frames and output characters.
    
    Args:
        log_probs: Log probabilities from the model [time, alphabet_size]
        decoded: Decoded sequence (label indices)
        input_length: Length of the input sequence
    """
    alphabet = "-abcdefghijklmnopqrstuvwxyz"
    probs = np.exp(log_probs[:input_length])
    
    plt.figure(figsize=(12, 6))
    plt.imshow(probs.T, aspect='auto', cmap='Blues')
    plt.colorbar(label='Probability')
    plt.xlabel('Time Frame')
    plt.ylabel('Character')
    
    # Mark the y-ticks with character labels
    plt.yticks(range(len(alphabet)), list(alphabet))
    
    # Highlight the decoded characters
    decoded_str = ''.join([alphabet[idx] for idx in decoded])
    plt.title(f'CTC Alignment - Decoded: "{decoded_str}"')
    
    # Draw horizontal lines to separate characters
    for i in range(1, len(alphabet)):
        plt.axhline(i-0.5, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train a simple handwriting recognition model
    model, losses = train_handwriting_model()
    
    # Test on an example and visualize
    test_input, test_target, test_input_len, test_target_len = generate_synthetic_data(
        batch_size=1, time_steps=50, feature_size=8, 
        num_classes=27, max_label_len=10
    )
    
    log_probs, _ = model.forward(test_input)
    ctc = CTCLoss()
    decoded = ctc.decode_greedy(log_probs, test_input_len)
    
    # Visualize the alignment
    visualize_ctc_alignment(log_probs[0], decoded[0], test_input_len[0])
