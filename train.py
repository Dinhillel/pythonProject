from transformers import AdamW
from model import T5Model
from dataset import TextDataset
from torch.utils.data import DataLoader
import torch

from torch.utils.data import DataLoader
from transformers import AdamW
from model import T5Model  # assuming you've defined T5Model
from datasets import load_dataset
import torch

# Assuming the TextDataset class is correctly defined in your code
train_dataset = TextDataset(tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize T5 model
model = T5Model()

# Set optimizer
optimizer = AdamW(model.model.parameters(), lr=1e-5)

# Train the modelimport torch
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
        x = self.fc5(x)  # Output layer (logits)
        return x


# Set input, hidden and output dimensions
input_dim = 5000  # Number of features (e.g., word vector size)
hidden_dim1 = 128  # First hidden layer dimension
hidden_dim2 = 64  # Second hidden layer dimension
hidden_dim3 = 32  # Third and fourth hidden layer dimension
output_dim = 2  # Number of classes (binary classification)

# Set device to CPU
device = torch.device('cpu')

# Initialize the model and move it to CPU
model = DeepANN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

# Print the model architecture
print(model)

# Example: Dummy input to test the model
example_input = torch.randn(1, input_dim).to(device)  # Single example with input_dim features

# Perform a forward pass through the network
logits = model(example_input)  # Logits are raw scores from the output layer

# Convert logits to probabilities using Softmax
softmax = nn.Softmax(dim=1)  # Softmax along the output dimension
probabilities = softmax(logits)

# Print the probabilities
print("Probabilities:", probabilities)
epochs = 3
for epoch in range(epochs):
    model.model.train()
    running_loss = 0.0

    for input_ids, labels in train_dataloader:
        optimizer.zero_grad()

        # Move the tensors to the correct device (if using GPU)
        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)

        # Forward pass through the model
        outputs = model.model(input_ids=input_ids, labels=labels)

        # Calculate loss
        loss = outputs.loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the loss after each epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}")

# Save the model after training
model.model.save_pretrained("t5_model")

