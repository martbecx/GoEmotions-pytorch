# train_dpo.py
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification

model_name = "monologg/bert-base-cased-goemotions-original"
# model_name = "Qwen/Qwen2-0.5B-Instruct"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

preference_example = Dataset.from_dict({"prompt": ["Please"], "chosen": ["blue"], "rejected": ["green"]})

print(f"preference_example: {preference_example[0]}")

training_args = DPOConfig(output_dir="QWEN", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=preference_example)
trainer.train()