# -*- coding: utf-8 -*-
"""
IMDB Sentiment Fine-Tuning with DistilBERT
Organized version
"""

# -----------------------------
# 1. Setup and Model Initialization
# -----------------------------
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_checkpoint = "distilbert-base-uncased"

# Load model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f">>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M")
print(f">>> BERT number of parameters: 110M")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# -----------------------------
# 2. Masked Token Prediction Example
# -----------------------------
import torch

text = "This is a greatest-when [MASK]."
inputs = tokenizer(text, return_tensors="pt")

# Get logits
token_logits = model(**inputs).logits

# Locate [MASK]
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]

# Get top 5 predictions
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")

# -----------------------------
# 3. Load IMDB Dataset
# -----------------------------
from datasets import load_dataset

imdb_dataset = load_dataset('imdb')

# Display sample
sample = imdb_dataset['unsupervised'].shuffle(seed=42).select(range(3))
for row in sample:
    print(f"\n>>> Review: {row['text']}")
    print(f">>> Label: {row['label']}")

# Unique labels
labels = imdb_dataset["test"].shuffle(seed=42).select(range(len(imdb_dataset["train"])))["label"]
unique_labels = set(labels)
print(unique_labels)

# -----------------------------
# 4. Tokenization
# -----------------------------
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# Tokenize dataset
tokenized_datasets = imdb_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])
tokenizer.model_max_length

# -----------------------------
# 5. Chunking and Grouping
# -----------------------------
chunk_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated_examples[list(examples.keys())[0]]) // chunk_size) * chunk_size
    result = {k: [t[i:i+chunk_size] for i in range(0, total_length, chunk_size)]
              for k, t in concatenated_examples.items()}
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# -----------------------------
# 6. Data Collator for MLM
# -----------------------------
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# -----------------------------
# 7. Whole Word Masking (Optional)
# -----------------------------
import collections
import numpy as np
from transformers import default_data_collator

wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)
        # Random mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels
    return default_data_collator(features)

# -----------------------------
# 8. Downsample Dataset for Quick Training
# -----------------------------
train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

# -----------------------------
# 9. Training Setup
# -----------------------------
from transformers import TrainingArguments, Trainer
import math

batch_size = 64
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# -----------------------------
# 10. Evaluation & Training
# -----------------------------
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train()

# -----------------------------
# 11. Fill-Mask Pipeline Example
# -----------------------------
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask",
    model="huggingface-course/distilbert-base-uncased-finetuned-imdb"
)

preds = mask_filler(text)
for pred in preds:
    print(f">>> {pred['sequence']}")
