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

# Train the model
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

