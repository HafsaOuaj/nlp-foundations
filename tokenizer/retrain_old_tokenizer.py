from pprint import pprint
from transformers import AutoTokenizer
from datasets import load_dataset

pprint("Tokenizer module loaded successfully.")

raw_datasets = load_dataset("code_search_net", "python",trust_remote_code=True)

pprint("Dataset loaded successfully.")
pprint(raw_datasets["train"])
pprint(raw_datasets["train"][123456]["whole_func_string"])
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )
training_corpus = get_training_corpus()
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
pprint(f"Old tokenizer tokens: {tokens}")
tokenizer = old_tokenizer.train_new_from_iterator(
    training_corpus, vocab_size=52000
)
tokenizer.save_pretrained("custom-gpt2-tokenizer")  
tokens = tokenizer.tokenize(example)
encoding = tokenizer(example)
pprint(f"the words ids of each tokenizer: {encoding.word_ids()}")
pprint(f"New tokenzer tokens: {tokens}")
pprint(f"Is the the tokenizer fast or not: {tokenizer.is_fast}")