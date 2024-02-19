import spacy
from datasets import load_dataset, get_dataset_config_names, load_dataset_builder, get_dataset_split_names
from torchtext.vocab import build_vocab_from_iterator

# define parameters
MIN_FREQ = 1

UNKNOWN_TOKEN = "<UNK>"
PADDING_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"

SPECIALS = [
    # unknown as default 0 index
    UNKNOWN_TOKEN,
    PADDING_TOKEN,
    EOS_TOKEN,
    SOS_TOKEN
]

MAX_SENTENCE_LEN = 200

training_set = load_dataset("wmt19", "zh-en", split="train[:100]")

zh_nlp = spacy.load("zh_core_web_sm")
en_nlp = spacy.load("en_core_web_sm")


def tokenize_example(example):
    en_tokens = [SOS_TOKEN] + [t.text for t in en_nlp.tokenizer(example['translation']['en'])][:MAX_SENTENCE_LEN] + [
        EOS_TOKEN]
    zh_tokens = [SOS_TOKEN] + [t.text for t in zh_nlp.tokenizer(example['translation']['zh'])][:MAX_SENTENCE_LEN] + [
        EOS_TOKEN]
    # if you return fields here, the map function on Dataset will add them to a new row. If you assign the returned
    # dataset then you have a new dataset with the new fields you want
    return {"en_tokens": en_tokens, "zh_tokens": zh_tokens}


training_set = training_set.map(tokenize_example)

print(type(training_set))

# build vocabulary here

en_vocab = build_vocab_from_iterator(training_set["en_tokens"], min_freq=MIN_FREQ, specials=SPECIALS)
zh_vocab = build_vocab_from_iterator(training_set["zh_tokens"], min_freq=MIN_FREQ, specials=SPECIALS)

# set default to pointing to unknown
en_vocab.set_default_index(0)
zh_vocab.set_default_index(0)


print(zh_vocab.lookup_indices(["牛马", "不存在的"]))
