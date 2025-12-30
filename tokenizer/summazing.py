from datasets import load_dataset
spanish_corpus = load_dataset("xnli", "es")
english_corpus = load_dataset("xnli", "en")

from pprint import pprint

pprint(spanish_corpus['train'][1])
pprint(english_corpus['train'][1])
def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> hypothesis: {example['hypothesis']}'")
        print(f"'>> premise: {example['premise']}'")


show_samples(spanish_corpus)
spanish_corpus['train'] = spanish_corpus['train'].shuffle(seed=42).select(range(1000))
spanish_corpus['test'] = spanish_corpus['test'].shuffle(seed=42).select(range(500))
spanish_corpus['validation'] = spanish_corpus['validation'].shuffle(seed=42).select(range(500))
english_corpus['train'] = english_corpus['train'].shuffle(seed=42).select(range(1000))
english_corpus['test'] = english_corpus['test'].shuffle(seed=42).select(range(500))
english_corpus['validation'] = english_corpus['validation'].shuffle(seed=42).select(range(500))
from datasets import DatasetDict,concatenate_datasets


books_dataset = DatasetDict()

for split in spanish_corpus.keys():
    books_dataset[split] = concatenate_datasets([spanish_corpus[split], english_corpus[split]])
    books_dataset[split] = books_dataset[split].shuffle(seed=42)
show_samples(books_dataset)
count_train_premise = [len(books_dataset['train'][i]["premise"]) for i in range(len(books_dataset['train']))]
count_train_hypothesis = [len(books_dataset['train'][i]["hypothesis"]) for i in range(len(books_dataset['train']))]
import matplotlib.pyplot as plt

plt.hist(count_train_premise, bins=30, alpha=0.5, label='Premise Lengths')
plt.hist(count_train_hypothesis, bins=30, alpha=0.5, label='Hypothesis Lengths')
plt.xlabel('Length (in characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Premise and Hypothesis Lengths in Training Set')
plt.legend()
plt.grid(True)
plt.show()

books_dataset = books_dataset.filter(lambda x:len(x['premise']) > 10 and len(x['hypothesis']) > 5)
from transformers import AutoTokenizer

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
inputs = tokenizer("I loved reading the Hunger Games!")
pprint(inputs)
tokenizer.convert_ids_to_tokens(inputs.input_ids)
max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["premise"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["hypothesis"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
tokenized_datasets
import evaluate 
rouge_score = evaluate.load("rouge")

generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"
scores = rouge_score.compute(
    predictions=[generated_summary], references=[reference_summary]
)
scores
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


pprint(three_sentence_summary(books_dataset["train"][100]["premise"]))
def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["premise"]]
    return metric.compute(predictions=summaries, references=dataset["hypothesis"])
books_dataset
import pandas as pd

score = evaluate_baseline(books_dataset["validation"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, round(score[rn] * 100, 2)) for rn in rouge_names)
pprint(rouge_dict)
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from transformers import Seq2SeqTrainingArguments

batch_size = 8
num_train_epochs = 1
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    #evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    #push_to_hub=True,
)
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}
features = [tokenized_datasets["train"][i] for i in range(2)]
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()
### Using the accelerator library to speed up training (optional)



