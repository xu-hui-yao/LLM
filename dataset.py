import datasets
from transformers import AutoTokenizer
import torch


class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, seq_length: int):
        """
        Args:
            split: dataset split ('train', 'test', 'validation')
            seq_length: sequence length
        """
        self.dataset = datasets.load_dataset('tiny_shakespeare', split=split)
        self.text = self.dataset['text'][0]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS to be PAD

        # Encode to token
        tokenized_text = self.tokenizer.encode(self.text, add_special_tokens=False)

        # Split to target length
        self.examples = [
            tokenized_text[i:i + seq_length]
            for i in range(0, len(tokenized_text) - seq_length + 1, seq_length)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return torch.tensor(input_ids, dtype=torch.long)
