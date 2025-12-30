# =========================
# Imports
# =========================
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import evaluate


# =========================
# Dataset & Labels
# =========================
def load_conll_dataset():
    raw_datasets = load_dataset("conll2003", trust_remote_code=True)

    print("Sample training example:", raw_datasets["train"][0])
    for split in raw_datasets:
        print(f"Number of examples in {split}: {len(raw_datasets[split])}")

    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_names = ner_feature.feature.names

    return raw_datasets, label_names


# =========================
# Tokenization & Label Alignment
# =========================
def label_reassignment(word_ids, labels):
    """
    Align word-level NER labels to token-level labels.
    """
    new_labels = []
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        else:
            new_labels.append(labels[word_id])
    return new_labels


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
    )

    all_labels = examples["ner_tags"]
    new_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = label_reassignment(word_ids, labels)
        new_labels.append(aligned_labels)

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# =========================
# Metrics
# =========================
def build_compute_metrics(label_names):
    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels = [
            [label_names[l] for l in label if l != -100]
            for label in labels
        ]
        true_predictions = [
            [
                label_names[p]
                for (p, l) in zip(prediction, label)
                if l != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


# =========================
# Training
# =========================
def train_ner_model():
    model_checkpoint = "bert-base-cased"

    # Load data
    raw_datasets, label_names = load_conll_dataset()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Tokenize dataset
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Label mappings
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    # Training arguments
    args = TrainingArguments(
        output_dir="bert-finetuned-conll2003-ner",
        #evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(label_names),
    )

    trainer.train()

    return args.output_dir


# =========================
# Inference
# =========================
def run_inference(model_checkpoint):
    token_classifier = pipeline(
        "token-classification",
        model=model_checkpoint,
        aggregation_strategy="simple",
    )

    sentence = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    predictions = token_classifier(sentence)

    print("\nInference result:")
    for p in predictions:
        print(p)


# =========================
# Main
# =========================
if __name__ == "__main__":
    checkpoint_path = train_ner_model()
    run_inference(checkpoint_path)
