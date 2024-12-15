import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import sigmoid
import csv
import random

class_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", 
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", 
    "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", 
    "sadness", "surprise", "neutral"
]

# Load pretrained model and tokenizer
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
model.eval()

# Function to load examples from devset
def load_examples(file_path, num_examples=10):
    examples = []
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        rows = list(reader)
        for i in range(num_examples):
            row = random.choice(rows)
            examples.append(row[0])
            data.append(tuple(row[1:]))
    return examples, data

# Function to get model outputs
def get_model_outputs(texts, model, tokenizer, device="cpu"):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = sigmoid(logits)
    results = []
    for i, text in enumerate(texts):
        top_probs, top_indices = torch.topk(probs[i], k=4)
        emotion_probs = [(class_labels[idx.item()], f"{prob.item():.2f}") for idx, prob in zip(top_indices, top_probs)]
        results.append({
            "text": text,
            "predicted_emotions": [emotion for emotion, _ in emotion_probs],
            "probabilities": [float(prob) for _, prob in emotion_probs],
            "emotion_probs": emotion_probs
        })
    return results

# Function to collect user feedback
def get_user_feedback(predictions):
    feedback_data = []
    for result in predictions:
        print(f"Text: {result['text']}")
        print(f"Predicted Emotions: {result['predicted_emotions']}")
        # print(f"Probabilities: {result['probabilities']}")
        while True:
            try:
                split_position = int(input("Enter the position to split between positive and negative (0-4): "))
                if 0 <= split_position <= 4:
                    break
                print("Please enter a number between 0 and 4.")
            except ValueError:
                print("Please enter a valid number.")
        feedback_data.append({
            "text": result["text"],
            "predicted_emotions": result["predicted_emotions"],
            "probabilities": result["probabilities"],
            "threshold": split_position
        })
    return feedback_data

# Function to write feedback to a new TSV file
def write_feedback_to_tsv(feedback_data, metadata, output_file):
    file_exists = False
    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(output_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        if not file_exists:
            writer.writerow(["Text", "Threshold"])
        for i in range(len(feedback_data)):
            feedback = feedback_data[i]
            data = metadata[i]
            writer.writerow([
                feedback["text"],
                data[0],
                data[1],
                feedback["threshold"]
            ])

# Main function
def main():
    dev_file = "data/original/dev.tsv"
    output_file = "data/thresh/feedback_results.tsv"
    num_iterations = 10

    examples, data = load_examples(dev_file, num_iterations)
    predictions = get_model_outputs(examples, model, tokenizer)
    feedback_data = get_user_feedback(predictions)
    write_feedback_to_tsv(feedback_data, data, output_file)

if __name__ == "__main__":
    main()