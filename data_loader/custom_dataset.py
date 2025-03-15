import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        tokens = self.tokenizer(self.sentences[item])
        indices = [self.vocab[token] for token in tokens]
        return torch.tensor(indices)


class CustomDatasetWithoutTokenization(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
