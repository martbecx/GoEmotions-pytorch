import torch
from transformers import BertTokenizer, BertForSequenceClassification
from trl import DPOTrainer, DPOConfig
from torch.nn.functional import sigmoid
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import Dataset

class DynamicThresholdModel(nn.Module):
    def __init__(self, base_model):
        super(DynamicThresholdModel, self).__init__()
        self.base_model = base_model
        self.config = base_model.config
        # Additional layer for predicting the dynamic threshold
        self.threshold_layer = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Predicts a threshold between 0 and 1
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the original logits [batch_size, num_emotions]
        original_logits = base_outputs.logits  # [batch_size, num_emotions]
        
        # Reshape logits to match what DPOTrainer expects
        # [batch_size, sequence_length, num_emotions]
        sequence_length = input_ids.size(1)
        reshaped_logits = original_logits.unsqueeze(1).expand(-1, sequence_length, -1)

        # Create an object that mimics the expected output format
        class OutputWrapper:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = hidden_states

        # Use the CLS token's hidden state for threshold prediction
        cls_hidden_state = base_outputs.hidden_states[-1][:, 0, :]
        predicted_threshold = self.threshold_layer(cls_hidden_state)

        return OutputWrapper(reshaped_logits, base_outputs.hidden_states)

    def get_emotion_logits(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Get the original emotion logits for prediction
        """
        base_outputs = self.base_model(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return base_outputs.logits

def predict_emotions_with_dynamic_threshold(texts, model, tokenizer, device="cpu"):
    """
    Predict emotions for a list of texts using the GoEmotions model with a dynamic threshold.
    """
    model.to(device)
    model.eval()

    # Tokenize and encode the input texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Get model outputs
    with torch.no_grad():
        logits = model.get_emotion_logits(**inputs)
        predicted_threshold = torch.tensor([0.5] * len(texts))  # Default threshold of 0.5
        probs = sigmoid(logits)  # Convert logits to probabilities

    # Rest of the function remains the same
    results = []
    for i, text in enumerate(texts):
        threshold = predicted_threshold[i].item()
        
        # Get top 4 predictions regardless of threshold
        top_probs, top_indices = torch.topk(probs[i], k=4)
        
        emotion_probs = [
            (class_labels[idx.item()], f"{prob.item():.2f}")
            for idx, prob in zip(top_indices, top_probs)
        ]

        results.append({
            "text": text,
            "predicted_emotions": [emotion for emotion, _ in emotion_probs],
            "probabilities": [float(prob) for _, prob in emotion_probs],
            "emotion_probs": emotion_probs,
            "threshold": threshold
        })
    return results

# User Feedback Workflow
def get_user_feedback(predictions):
    """
    Collect user feedback for thresholds and split results into positive and negative.

    Args:
        predictions (list): List of predictions containing text, predicted emotions, probabilities, and dynamic thresholds.

    Returns:
        list: Dataset entries with input, positive and negative splits, and thresholds.
    """
    feedback_data = []

    for result in predictions:
        print(f"Text: {result['text']}")
        print(f"Predicted Emotions: {result['predicted_emotions']}")
        print(f"Probabilities: {result['probabilities']}")
        print(f"Dynamic Threshold: {result['threshold']}")

        # Get user input for the split position
        while True:
            try:
                split_position = int(input("Enter the position to split between positive and negative (1-4): "))
                if 1 <= split_position <= 4:
                    break
                print("Please enter a number between 1 and 4. This will be the position to split between positive and negative where e.g. 1 would result in the first emotion being positive and the rest being negative.")
            except ValueError:
                print("Please enter a valid number.")

        # Split results based on position
        positive_results = [
            (emotion, prob)
            for emotion, prob in zip(result['predicted_emotions'][:split_position], 
                                   result['probabilities'][:split_position])
        ]
        negative_results = [
            (emotion, prob)
            for emotion, prob in zip(result['predicted_emotions'][split_position:], 
                                   result['probabilities'][split_position:])
        ]

        feedback_data.append({
            "input": result["text"],
            "chosen": positive_results,
            "rejected": negative_results,
        })

    print(f"Feedback Data: {feedback_data}")
    return feedback_data

# Prepare dataset for DPOTrainer
def prepare_dpo_dataset(feedback_data, tokenizer):
    """
    Convert user feedback into a format suitable for DPO training.
    """
    dataset = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    for feedback in feedback_data:
        # Store the original text as prompt
        dataset["prompt"].append(feedback["input"])
        
        # Format chosen predictions as a single string with probabilities
        chosen_text = " | ".join([
            f"{emotion} ({prob:.2f})"
            for emotion, prob in feedback["chosen"]
        ])
        
        # Format rejected predictions as a single string with probabilities
        rejected_text = " | ".join([
            f"{emotion} ({prob:.2f})"
            for emotion, prob in feedback["rejected"]
        ])
        
        dataset["chosen"].append(chosen_text)
        dataset["rejected"].append(rejected_text)

    # Convert to HuggingFace Dataset
    # Let DPOTrainer handle the tokenization
    print(f"dataset: {dataset}")
    return Dataset.from_dict(dataset)

# Fine-tune with DPOTrainer
def fine_tune_with_dpo(feedback_dataset, model, tokenizer, output_dir="dpo-finetuned"):
    """
    Fine-tune the model with DPO training using user feedback.
    """
    # Create a simple data collator
    def custom_data_collator(features):
        # Tokenize all texts
        prompts = [f["prompt"] for f in features]
        chosen = [f["chosen"] for f in features]
        rejected = [f["rejected"] for f in features]
        
        # Tokenize with padding
        prompt_tokens = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        chosen_tokens = tokenizer(
            chosen,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        rejected_tokens = tokenizer(
            rejected,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }

    # Configure DPO training
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        max_length=512,
        max_prompt_length=128,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        logging_dir=f"{output_dir}/logs",
    )

    training_args = DPOConfig(output_dir=output_dir, logging_steps=10)
    trainer = DPOTrainer(model=model, args=dpo_config, processing_class=tokenizer, train_dataset=feedback_dataset)

    # Train the model
    trainer.train()


# Load pretrained model and tokenizer
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = BertTokenizer.from_pretrained(model_name)
base_model = BertForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
model = DynamicThresholdModel(base_model)

# GoEmotions class labels
class_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", 
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", 
    "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", 
    "sadness", "surprise", "neutral"
]

input_texts = [
    "You know the answer man, you are programmed to capture those codes they send you, don’t avoid them!",
    "I've never been this sad in my life!",
    "The economy is heavily controlled and subsidized by the government. In any case, I was poking at the lack of nuance in US politics today.",
    "He could have easily taken a real camera from a legitimate source and change the price in Word/Photoshop and then print it out.",
]

# Main Workflow
if __name__ == "__main__":
    # Initial predictions
    predictions = predict_emotions_with_dynamic_threshold(input_texts, model, tokenizer)

    # User feedback loop
    for loop in range(5):  # Repeat for 5 iterations
        print(f"\n==== Feedback Loop {loop + 1} ====\n")
        
        # Collect user feedback
        feedback_data = get_user_feedback(predictions)
        
        # Prepare DPO training dataset
        feedback_dataset = prepare_dpo_dataset(feedback_data, tokenizer)
        
        # Fine-tune the model using the DPOTrainer
        fine_tune_with_dpo(feedback_dataset, model, tokenizer)
        
        # Re-run predictions with updated model
        predictions = predict_emotions_with_dynamic_threshold(input_texts, model, tokenizer)

