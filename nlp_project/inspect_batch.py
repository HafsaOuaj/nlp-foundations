import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset_text import TextDataset  # your DatasetText class

def inspect_batch(texts, max_length=16, batch_size=2):
    """
    Inspect tokenized batches.
    
    Args:
        texts (list[str]): List of raw text strings
        max_length (int): Max token sequence length
        batch_size (int): Batch size for DataLoader
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = TextDataset(
        texts, 
        max_length=max_length, 
        tokenizer=tokenizer, 
        pad_token_id=tokenizer.pad_token_id
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    print("Batch input_ids shape:", input_ids.shape)
    print("Batch attention_mask shape:", attention_mask.shape)
    print("\n--- Inspecting first batch ---")
    for i in range(input_ids.size(0)):
        ids = input_ids[i]
        mask = attention_mask[i]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"\nSample {i}:")
        print("Tokens:", tokens)
        print("Attention mask:", mask.tolist())
        print("Decoded text:", tokenizer.decode(ids, skip_special_tokens=True))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a batch of tokenized texts")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for DataLoader")
    args = parser.parse_args()

    # Example texts
    texts = [
        "Hello, how are you?",
        "This is a test sentence for the TextDataset class.",
        "Transformers library makes it easy to work with pre-trained models.",
        "Batching multiple sentences is simple once you understand padding.",
    ]

    inspect_batch(texts, max_length=args.max_length, batch_size=args.batch_size)
