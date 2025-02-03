import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define the Deep Neural Network (ANN) with 5 layers
class DeepANN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(DeepANN, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)  # Third hidden layer
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim3)  # Fourth hidden layer
        self.fc5 = nn.Linear(hidden_dim3, output_dim)  # Output layer

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through each layer with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation function here for output layer
        return x


# Set input, hidden and output dimensions
input_dim = 5000  # Example: Number of features (for text, it could be the word count or vector size)
hidden_dim1 = 128  # First hidden layer dimension
hidden_dim2 = 64  # Second hidden layer dimension
hidden_dim3 = 32  # Third and fourth hidden layer dimension
output_dim = 2  # Number of classes (for binary classification)

# Initialize the model
model = DeepANN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)

# Print the model architecture
print(model)

