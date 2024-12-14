from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load pretrained GoEmotions model and tokenizer
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

class_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", 
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", 
    "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", 
    "sadness", "surprise", "neutral"
]

# Test the model with a sample text
texts = [
    "Hey that's a thought! Maybe we need [NAME] to be the celebrity vaccine endorsement!",
    "it’s happened before?! love my hometown of beautiful new ken 😂😂",
    "I love you, brother.",
    "Troll, bro. They know they're saying stupid shit. The motherfucker does nothing but stink up libertarian subs talking shit",
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")  # Tokenize the input text
    outputs = model(**inputs)  # Get model outputs
    logits = outputs.logits  # Get the logits (raw scores)
    logits = 1 / (1 + torch.exp(-logits.detach()))

    # Decode the predictions
    values, indices = torch.sort(logits, descending=True, dim=1, out=None)
    print(text)

    flag = True
    for value, indice in zip(torch.flatten(values), torch.flatten(indices)):
        if value.item() < 0.3 and flag:
            print('---------------------------------------------')
            flag = False
        print(class_labels[indice], value.item())
    print()