import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from trl import PPOTrainer, PPOConfig
from torch.nn.functional import sigmoid
from datasets import Dataset

class DynamicThresholdModel(nn.Module):
    def __init__(self, base_model):
        super(DynamicThresholdModel, self).__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.threshold_layer = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        base_outputs = self.base_model(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        original_logits = base_outputs.logits
        cls_hidden_state = base_outputs.hidden_states[-1][:, 0, :]
        predicted_threshold = self.threshold_layer(cls_hidden_state)
        return original_logits, predicted_threshold

    def get_emotion_logits(self, input_ids, attention_mask=None, token_type_ids=None):
        base_outputs = self.base_model(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return base_outputs.logits

def predict_emotions_with_dynamic_threshold(texts, model, tokenizer, device="cpu"):
    model.to(device)
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits, predicted_threshold = model(**inputs)
        probs = sigmoid(logits)
    results = []
    for i, text in enumerate(texts):
        threshold = predicted_threshold[i].item()
        top_probs, top_indices = torch.topk(probs[i], k=4)
        emotion_probs = [(class_labels[idx.item()], f"{prob.item():.2f}") for idx, prob in zip(top_indices, top_probs)]
        results.append({
            "text": text,
            "predicted_emotions": [emotion for emotion, _ in emotion_probs],
            "probabilities": [float(prob) for _, prob in emotion_probs],
            "emotion_probs": emotion_probs,
            "threshold": threshold
        })
    return results

def get_user_feedback(predictions):
    feedback_data = []
    for result in predictions:
        print(f"Text: {result['text']}")
        print(f"Predicted Emotions: {result['predicted_emotions']}")
        print(f"Probabilities: {result['probabilities']}")
        print(f"Dynamic Threshold: {result['threshold']}")
        while True:
            try:
                split_position = int(input("Enter the position to split between positive and negative (1-4): "))
                if 1 <= split_position <= 4:
                    break
                print("Please enter a number between 1 and 4.")
            except ValueError:
                print("Please enter a valid number.")
        positive_results = [(emotion, prob) for emotion, prob in zip(result['predicted_emotions'][:split_position], result['probabilities'][:split_position])]
        negative_results = [(emotion, prob) for emotion, prob in zip(result['predicted_emotions'][split_position:], result['probabilities'][split_position:])]
        feedback_data.append({
            "input": result["text"],
            "chosen": positive_results,
            "rejected": negative_results,
        })
        print(f"Positive Results: {positive_results}")
        print(f"Negative Results: {negative_results}\n")
    print(f"Feedback Data: {feedback_data}")
    return feedback_data

def prepare_ppo_dataset(feedback_data, tokenizer):
    dataset = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for feedback in feedback_data:
        dataset["prompt"].append(feedback["input"])
        chosen_text = " | ".join([f"{emotion} ({prob:.2f})" for emotion, prob in feedback["chosen"]])
        rejected_text = " | ".join([f"{emotion} ({prob:.2f})" for emotion, prob in feedback["rejected"]])
        dataset["chosen"].append(chosen_text)
        dataset["rejected"].append(rejected_text)
    return Dataset.from_dict(dataset)

def fine_tune_with_ppo(feedback_dataset, model, tokenizer, output_dir="ppo-finetuned"):
    def custom_data_collator(features):
        prompts = [f["prompt"] for f in features]
        chosen = [f["chosen"] for f in features]
        rejected = [f["rejected"] for f in features]
        prompt_tokens = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        chosen_tokens = tokenizer(chosen, padding=True, truncation=True, max_length=512, return_tensors="pt")
        rejected_tokens = tokenizer(rejected, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return {
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }
    ppo_config = PPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        logging_dir=f"{output_dir}/logs",
    )
    trainer = PPOTrainer(
        reward_model=model,
        policy=None,
        ref_policy=None,
        config=ppo_config,
        train_dataset=feedback_dataset,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
    )
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

# Example usage
input_texts = [
    "I love you, brother.",
    "Troll, bro. They know they're saying stupid shit.",
]

# Initial predictions
predictions = predict_emotions_with_dynamic_threshold(input_texts, model, tokenizer)

# User feedback loop
for loop in range(5):
    print(f"\n==== Feedback Loop {loop + 1} ====\n")
    feedback_data = get_user_feedback(predictions)
    feedback_dataset = prepare_ppo_dataset(feedback_data, tokenizer)
    fine_tune_with_ppo(feedback_dataset, model, tokenizer)
    predictions = predict_emotions_with_dynamic_threshold(input_texts, model, tokenizer)