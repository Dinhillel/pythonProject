from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")


# Function to generate summary
def generate_summary(input_text):
    model.eval()  # Set model to evaluation mode
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    summary_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Dataset class
class TextDataset(Dataset):
    def __init__(self, tokenize, max_length=512, dataset_name="xsum", split="train"):
        # Load dataset
        self.dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)  # Add trust_remote_code
        self.tokenizer = tokenize
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        article = self.dataset[idx]['document']  # Change 'article' to 'document' for XSum
        highlights = self.dataset[idx]['summary']  # Change 'highlights' to 'summary' for XSum

        # Tokenize the article and highlights
        inputs = self.tokenizer(article, padding='max_length', truncation=True, max_length=self.max_length,
                                return_tensors="pt")
        labels = self.tokenizer(highlights, padding='max_length', truncation=True, max_length=self.max_length,
                                return_tensors="pt")

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        labels_ids = labels['input_ids'].squeeze(0)

        # Replace padding tokens in labels with -100 (important for loss calculation)
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels_ids


# Set device (use GPU if available)
device = torch.device("cpu")
model.to(device)

# Prepare the dataset
train_dataset = TextDataset(tokenizer)

# Set up DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(3):  # Train for 3 epochs
    model.train()
    running_loss = 0.0

    for input_ids, attention_mask, labels_ids in train_dataloader:
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_ids = labels_ids.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}")

# Save the trained model
model.save_pretrained("trained_t5_model")
