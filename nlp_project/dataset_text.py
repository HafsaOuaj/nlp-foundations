import torch
from transformers import AutoTokenizer



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts,max_length , tokenizer,pad_token_id, pad_to_max_length=True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_to_max_length = pad_to_max_length

    def __len__(self):
        return len(self.texts)
    

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            padding='max_length' if self.pad_to_max_length else False,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        tokens = encoded['input_ids'].squeeze(0)
        mask = encoded['attention_mask'].squeeze(0)
        return {"input_ids": tokens, "attention_mask": mask}

    
if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    texts = [
        "Hello, how are you?",
        "This is a test sentence for the TextDataset class.",
        "Transformers library makes it easy to work with pre-trained models.",
    ]
    dataset = TextDataset(texts, max_length=16, tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)
    for i in range(len(dataset)):
        item = dataset[i]
        print(f"Text {i}: {texts[i]}")
        print(f"Tokens: {item['input_ids']}")
        print(f"Mask: {item['attention_mask']}")
