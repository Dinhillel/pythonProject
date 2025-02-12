from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
import torch

# Load the dataset (CNN/
dataset = load_dataset("cnn_dailymail", "1.0.0", split="train")

# Define the custom Dataset
class TextDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, dataset_name="cnn_dailymail", split="train"):
        self.dataset = load_dataset(dataset_name, '1.0.0', split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the article and the highlights (summary)
        article = self.dataset[idx]['article']
        highlights = self.dataset[idx]['highlights']

        # Tokenize the article and highlights
        inputs = self.tokenizer(article, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(highlights, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        # Extract input and label ids (remove batch dimension)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()  # Get attention mask
        labels_ids = labels['input_ids'].squeeze()

        return input_ids, attention_mask, labels_ids

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)

# Create the training dataset
train_dataset = TextDataset(tokenizer)

# Initialize the T5 model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

# Set device to CPU
device = torch.device("cpu")  # No need for GPU here
model.to(device)

# Create the DataLoader for the training data
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Set optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Set the number of epochs
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for input_ids, attention_mask, labels_ids in train_dataloader:
        optimizer.zero_grad()

        # Move tensors to CPU (you already defined the device as CPU)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_ids = labels_ids.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_ids)

        # Calculate loss
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

# Save the trained model
model.save_pretrained("trained_t5_model")

# Function to generate a summary from a new input article after training
def generate_summary(input_text):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length').to(device)

    summary_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example to test the trained model
input_article = "Your example article text goes here. This should be a long article to test the summarization model."
summary = generate_summary(input_article)
print("Generated Summary:", summary)
