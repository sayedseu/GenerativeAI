import spacy
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

dataset = [
    (1, "Introduction to NLP"),
    (2, "Basics of PyTorch"),
    (1, "NLP Techniques for Text Classification"),
    (3, "Named Entity Recognition with PyTorch"),
    (3, "Sentiment Analysis using PyTorch"),
    (3, "Machine Translation with PyTorch"),
    (1, " NLP Named Entity,Sentiment Analysis,Machine Translation "),
    (1, " Machine Translation with NLP "),
    (1, " Named Entity vs Sentiment Analysis  NLP ")]

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


my_iterator = yield_tokens(dataset)

vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def get_tokenized_sentence_and_indices(iterator):
    sentence = next(iterator)
    indices = [vocab[token] for token in sentence]
    return sentence, indices


tokenized_sentence, token_indices = get_tokenized_sentence_and_indices(my_iterator)

print("Tokenized Sentence:", tokenized_sentence)
print("Token Indices:", token_indices)
