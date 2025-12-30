# -*- coding: utf-8 -*-

# ==============================
# 1️⃣ Imports
# ==============================
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
import evaluate

# ==============================
# 2️⃣ Load Dataset
# ==============================
raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets['train']
raw_validation_dataset = raw_datasets['validation']
raw_test_dataset = raw_datasets['test']

# ==============================
# 3️⃣ Load Tokenizer
# ==============================
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ==============================
# 4️⃣ Preprocess / Tokenize
# ==============================
def preprocess_data(sample):
    return tokenizer(
        sample["sentence1"],
        sample["sentence2"],
        padding=True,
        truncation=True
    )

tokenized_datasets = raw_datasets.map(preprocess_data, batched=True)

# ==============================
# 5️⃣ Data Collator
# ==============================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Clean up columns and set format
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# ==============================
# 6️⃣ Create DataLoaders
# ==============================
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=10,
    collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator
)

# Quick check
for batch in train_dataloader:
    print({k: v.shape for k, v in batch.items()})
    break

# ==============================
# 7️⃣ Load Model
# ==============================
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

# Test forward pass
outputs = model(**batch)
print("Loss:", outputs.loss, "Logits shape:", outputs.logits.shape)

# ==============================
# 8️⃣ Optimizer
# ==============================
optimizer = AdamW(model.parameters(), lr=5e-5)

# ==============================
# 9️⃣ Device setup
# ==============================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print("Using device:", device)

# ==============================
# 10️⃣ Accelerator preparation
# ==============================
accelerator = Accelerator()
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

# ==============================
# 11️⃣ Scheduler
# ==============================
num_epochs = 3
num_training_steps = len(train_dl) * num_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
print("Total training steps:", num_training_steps)

# Progress bar
progress_bar = tqdm(range(num_training_steps))

# ==============================
# 12️⃣ Training Loop
# ==============================
model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# ==============================
# 13️⃣ Evaluation
# ==============================
metric = evaluate.load("glue", "mrpc")
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

results = metric.compute()
print("Evaluation results:", results)
