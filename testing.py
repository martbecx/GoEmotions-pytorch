from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pretrained GoEmotions model and tokenizer
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Test the model with a sample text
text = "I want to chuck a beer?"
inputs = tokenizer(text, return_tensors="pt")  # Tokenize the input text
outputs = model(**inputs)  # Get model outputs
logits = outputs.logits  # Get the logits (raw scores)

# Decode the predictions
predicted_class = torch.argmax(logits, dim=1)
print(f"Predicted class: {predicted_class.item()}")
