import torch
import torch.nn as nn

# ANN model for NLP
class DeepANN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(DeepANN, self).__init__()

        # Fully Connected Layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim3)
        self.fc5 = nn.Linear(hidden_dim3, output_dim)  # Output layer

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation function here
        return x  # Raw logits (softmax should be applied outside when needed)

# Model parameters
input_dim = 5000  # Number of features in text representation
hidden_dim1 = 128
hidden_dim2 = 64
hidden_dim3 = 32
output_dim = 2  # Binary classification

# again set the model
model = DeepANN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)

# Example: Feeding text vector into the model
example_input = torch.randn(1, input_dim)  # Example input vector
logits = model(example_input)  # Get raw logits

# Convert logits to probabilities using softmax
softmax = nn.Softmax(dim=1)
probabilities = softmax(logits)

print("Logits:", logits)
print("Probabilities:", probabilities)
