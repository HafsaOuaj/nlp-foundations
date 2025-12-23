from transformers import AutoTokenizer
from pprint import pprint
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example  ="HI THERE THIS IS A TEST!"
tokens = tokenizer.tokenize(example)
pprint("Bert tokenizer loaded successfully.")
pprint(f"{tokenizer(example).word_ids()}")
start, end = tokenizer(example).word_to_chars(2)
example[start:end]



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
pprint(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))