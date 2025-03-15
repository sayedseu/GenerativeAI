import torch.nn.utils.rnn
import torchtext.vocab
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer

from custom_dataset import CustomDataset, CustomDatasetWithoutTokenization

sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

tokenizer = get_tokenizer("basic_english")

vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, sentences))

custom_dataset = CustomDatasetWithoutTokenization(sentences)


def collate_function(batch):
    batch_indices = []
    for sample in batch:
        tokens = tokenizer(sample)
        indices = [vocab[token] for token in tokens]
        batch_indices.append(torch.tensor(indices))

    padded_batch = torch.nn.utils.rnn.pad_sequence(batch_indices, batch_first=True, padding_value=0)
    return padded_batch


dataloader = DataLoader(dataset=custom_dataset, batch_size=2, shuffle=False, collate_fn=collate_function)

for batch in dataloader:
    print(batch)
