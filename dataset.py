from datasets import load_dataset
from transformers import T5Tokenizer
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, dataset_name="cnn_dailymail", split="train"):
        # Load the dataset with a version (e.g., '1.0.0')
        self.dataset = load_dataset(dataset_name, '1.0.0', split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load the text (article) and highlights (summary)
        article = self.dataset[idx]['article']
        highlights = self.dataset[idx]['highlights']

        # Tokenize the article and highlights
        inputs = self.tokenizer(article, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(highlights, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        # Extract input and label ids (removing batch dimension)
        input_ids = inputs['input_ids'].squeeze()  # Remove batch dimension
        labels_ids = labels['input_ids'].squeeze()

        return input_ids, labels_ids

# Load the T5 tokenizer with legacy=False
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)

# Create the dataset for training
train_dataset = TextDataset(tokenizer)
