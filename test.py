from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig
import torch

# Model and tokenizer
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)

# Dataset
preference_example = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": ["neutral", "neutral"],
    "rejected": ["admiration", "admiration"]
})

# Custom collator
def custom_data_collator(features):
    # Debug: Inspect the raw features
    print("Features received by custom_data_collator:", features)

    def clean_input(input_list):
        """Remove None values from input lists."""
        return [token for token in input_list if token is not None]

    # Clean the features by removing None values in the tokenized input lists
    for feature in features:
        for key in ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"]:
            if key in feature:
                feature[key] = clean_input(feature[key])

    # Ensure tokenization of the prompt and generation of attention mask
    prompt_input_ids = [f["prompt_input_ids"] for f in features]
    prompt_attention_mask = [
        [1] * len(input_ids) for input_ids in prompt_input_ids  # 1's for all valid tokens
    ]
    
    # Process the 'chosen' and 'rejected' inputs (already handled)
    chosen_input_ids = [f["chosen_input_ids"] for f in features]
    chosen_attention_mask = [
        [1] * len(input_ids) for input_ids in chosen_input_ids
    ]
    
    rejected_input_ids = [f["rejected_input_ids"] for f in features]
    rejected_attention_mask = [
        [1] * len(input_ids) for input_ids in rejected_input_ids
    ]

    # Debug: Print processed input data
    print("Processed input data:")
    print("Prompt input ids:", prompt_input_ids)
    print("Prompt attention mask:", prompt_attention_mask)
    print("Chosen input ids:", chosen_input_ids)
    print("Chosen attention mask:", chosen_attention_mask)
    print("Rejected input ids:", rejected_input_ids)
    print("Rejected attention mask:", rejected_attention_mask)

    # Return final tensors with the correct attention masks
    return {
        "prompt_input_ids": torch.tensor(prompt_input_ids),
        "prompt_attention_mask": torch.tensor(prompt_attention_mask),
        "chosen_input_ids": torch.tensor(chosen_input_ids),
        "chosen_attention_mask": torch.tensor(chosen_attention_mask),
        "rejected_input_ids": torch.tensor(rejected_input_ids),
        "rejected_attention_mask": torch.tensor(rejected_attention_mask),
    }

# Training configuration
training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=preference_example,
    data_collator=custom_data_collator,
)

# Train
trainer.train()
