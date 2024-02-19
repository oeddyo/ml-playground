import spacy
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator


class TranslationData:
    def __init__(self, min_freq=1, max_sentence_len=200, src_lang="en", dest_lang="zh"):
        self.min_freq = min_freq
        self.max_sentence_len = max_sentence_len
        self.src_lang = src_lang
        self.dest_lang = dest_lang

        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unknown_token = "<UNK>"
        self.padding_token = "<PADDING>"

        self.special_tokens = [
            # unknown as default 0 index
            self.unknown_token,
            self.padding_token,
            self.eos_token,
            self.sos_token
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
            "src_tokens": [self.sos_token] + en_tokens + [self.eos_token],
            "dest_tokens": [self.sos_token] + zh_tokens + [self.eos_token]
        }

    def _vectorize(self, example):
        return {
            "src_ids": self.src_vocab.lookup_indices(example["src_tokens"]),
            "dest_ids": self.dest_vocab.lookup_indices(example["dest_tokens"])
        }


t = TranslationData(min_freq=1)
print(t.get_datasets()[0][0])
