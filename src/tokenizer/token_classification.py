from transformers import pipeline
from pprint import pprint

token_classfier = pipeline("token-classification")
example = "Hafsa works at Microsoft in Seattle."
results = token_classfier(example)
pprint(results)
from transformers import AutoTokenizer,AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "Hafsa works at Microsoft in Seattle."
tokens = tokenizer(example, return_tensors="pt")
outputs = model(**tokens)
pprint(f"the shape of outputs: {outputs.logits.shape}")
pprint(f"the shape of the input{tokens['input_ids'].shape}")
print(f"the shape of outputs: {outputs.logits.shape}")
print(f"the shape of the input{tokens['input_ids'].shape}")
import torch

probabs = torch.nn.functional.softmax(outputs.logits, dim=-1)
preds = probabs.argmax(dim=2)
print(f"the shape of preds: {preds.shape} \npreds: {preds}")
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)
results = []

inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

pprint(results)