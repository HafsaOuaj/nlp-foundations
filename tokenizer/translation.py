# =========================
# 1. Imports
# =========================
import numpy as np
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate


# =========================
# 2. Constants & Config
# =========================
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-fr"
MAX_INPUT_LENGTH = 128
DATASET_NAME = "europarl_bilingual"
DATASET_CONFIG = "en-fr"


# =========================
# 3. Dataset Loading
# =========================
def load_and_split_dataset():
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train[:1000]",
        trust_remote_code=True,
    )
    return dataset.train_test_split(test_size=0.2)


# =========================
# 4. Tokenizer & Preprocessing
# =========================
def preprocess_function(examples, tokenizer):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )
    return model_inputs


# =========================
# 5. Metrics (SacreBLEU)
# =========================
metric = evaluate.load("sacrebleu")


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True
    )

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    return {"bleu": result["score"]}


# =========================
# 6. Main Training Script
# =========================
def main():
    # Load dataset
    dataset = load_and_split_dataset()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Tokenize dataset
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Load model via pipeline (translation)
    translator = pipeline(
        "translation",
        model=MODEL_CHECKPOINT,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=translator.model,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="marian-finetuned-en-fr",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=translator.model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    # Evaluate before training
    print("Initial evaluation:")
    trainer.evaluate(max_length=MAX_INPUT_LENGTH)

    # Train
    trainer.train()

    # Final evaluation
    print("Final evaluation:")
    trainer.evaluate(max_length=MAX_INPUT_LENGTH)


# =========================
# 7. Entry Point
# =========================
if __name__ == "__main__":
    main()
