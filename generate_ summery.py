from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# difine the cpu train
device = torch.device("cpu")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")


# function
def generate_summary(input_text, summary_ids=None):
    model.eval()

    # replace from text to token
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length').to(
        device)


    summary_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150)

    # and token to text
    summary = tokenizer.decode(summary[0], skip_special_tokens=True)from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Define the CPU device
device = torch.device("cpu")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")


# Function to summarize text
def generate_summary(input_text):
    model.eval()
    print("Input Text:", input_text)  # Debug input text

    # Tokenize and move inputs to the correct device
    inputs = tokenizer(
        input_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length'
    ).to(device)

    print("Tokenized Inputs:", inputs)  # Debug the tokenized inputs

    # Generate summary IDs
    summary_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=150,
        num_beams=2,  # Beam search for better summaries
        early_stopping=True  # Stop when output seems complete
    )
    print("Summary IDs:", summary_ids)  # Debug the generated ID tensors

    # Decode summary IDs to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("Decoded Summary:", summary)  # Debug the final summary
    return summary


# An example input article
input_article = """
The United Nations has been involved in peacekeeping operations around the world for decades. Their role is to help maintain peace and security in conflict regions, assist in rebuilding nations after war, and support humanitarian efforts. Over the years, their presence in many countries has been pivotal in preventing further violence and stabilizing troubled areas.
"""
try:
    # Generate summary
    summary = generate_summary(input_article)

    # Print the summary
    print("Generated Summary:", summary)

    # Save the text to a file
    with open("generated_summary.txt", "w") as f:
        f.write(summary)

except Exception as e:
    print("An error occurred:", e)

    return summary


input_article = """
The United Nations has been involved in peacekeeping operations around the world for decades. Their role is to help maintain peace and security in conflict regions, assist in rebuilding nations after war, and support humanitarian efforts. Over the years, their presence in many countries has been pivotal in preventing further violence and stabilizing troubled areas.
"""
summary = generate_summary(input_article)

# print thr summery
print("Generated Summary:", summary)

#save the text
with open("generated_summary.txt", "w") as f:
    f.write(summary)
