import spacy
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn

UNKNOWN_TOKEN = "<UNK>"
PADDING_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"


class TranslationData:
    def __init__(self, min_freq=1, max_sentence_len=200, src_lang="en", dest_lang="zh"):
        self.min_freq = min_freq
        self.max_sentence_len = max_sentence_len
        self.src_lang = src_lang
        self.dest_lang = dest_lang

        self.special_tokens = [
            UNKNOWN_TOKEN,
            PADDING_TOKEN,
            EOS_TOKEN,
            SOS_TOKEN
        ]

        # download dataset
        training_set = load_dataset("wmt19", "zh-en", split="train[:100]", trust_remote_code=True)
        validation_set = load_dataset("wmt19", "zh-en", split="validation[:100]", trust_remote_code=True)

        self.src_nlp = spacy.load("en_core_web_sm")
        self.dest_nlp = spacy.load("zh_core_web_sm")

        training_set = training_set.map(self._tokenize)
        validation_set = validation_set.map(self._tokenize)

        self.src_vocab = build_vocab_from_iterator(training_set["src_tokens"], min_freq=self.min_freq,
                                                   specials=self.special_tokens)
        self.dest_vocab = build_vocab_from_iterator(training_set["dest_tokens"], min_freq=self.min_freq,
                                                    specials=self.special_tokens)

        self.src_vocab.set_default_index(0)
        self.dest_vocab.set_default_index(0)

        columns_to_torch = ["src_ids", "dest_ids"]
        self.training_set = training_set.map(self._vectorize).with_format(
            type="torch", columns=columns_to_torch, output_all_columns=True
        )
        self.validation_set = validation_set.map(self._vectorize).with_format(
            type="torch", columns=columns_to_torch, output_all_columns=True
        )

    def get_vocab(self):
        return self.src_vocab, self.dest_vocab

    def get_datasets(self):
        return self.training_set, self.validation_set

    def _tokenize(self, example):
        en_tokens = [t.text for t in self.src_nlp.tokenizer(example['translation'][self.src_lang])][
                    :self.max_sentence_len]
        zh_tokens = [t.text for t in self.dest_nlp.tokenizer(example['translation'][self.dest_lang])][
                    :self.max_sentence_len]

        # if you return fields here, the map function on Dataset will add them to a new row. If you assign the returned
        # dataset then you have a new dataset with the new fields you want
        return {
            "src_tokens": [SOS_TOKEN] + en_tokens + [EOS_TOKEN],
            "dest_tokens": [SOS_TOKEN] + zh_tokens + [EOS_TOKEN]
        }

    def _vectorize(self, example):
        return {
            "src_ids": self.src_vocab.lookup_indices(example["src_tokens"]),
            "dest_ids": self.dest_vocab.lookup_indices(example["dest_tokens"])
        }


class TDataset(Dataset):

    def __init__(self, translation_data):
        self.translation_data = translation_data

    def __getitem__(self, item):
        return self.translation_data[item]

    def __len__(self):
        return len(self.translation_data)


training_set, validation_set = TranslationData().get_datasets()


def collate_fn(batch):
    batch_src = [e["src_ids"] for e in batch]
    batch_src = nn.utils.rnn.pad_sequence(batch_src, batch_first=True, padding_value=1)

    batch_dest = [e["dest_ids"] for e in batch]
    batch_dest = nn.utils.rnn.pad_sequence(batch_dest, batch_first=True, padding_value=1)
    return batch_src, batch_dest


tl = DataLoader(TDataset(training_set), batch_size=64, collate_fn=collate_fn)

for b in tl:
    print(b[0].shape, b[1].shape)
