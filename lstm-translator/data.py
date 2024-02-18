import spacy
from datasets import load_dataset, get_dataset_config_names, load_dataset_builder, get_dataset_split_names

# define parameters
UNKNOWN_TOKEN = "<UNK>"
PADDING_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"

MAX_SENTENCE_LEN = 200

training_set = load_dataset("wmt19", "zh-en", split="train[:100]")

print(training_set[0]["translation"]["zh"])

zh_nlp = spacy.load("zh_core_web_sm")
en_nlp = spacy.load("en_core_web_sm")


def tokenize_example(example):
    en_tokens = [SOS_TOKEN] + [t.text for t in en_nlp.tokenizer(example['translation']['en'])][:MAX_SENTENCE_LEN] + [
        EOS_TOKEN]
    zh_tokens = [SOS_TOKEN] + [t.text for t in zh_nlp.tokenizer(example['translation']['zh'])][:MAX_SENTENCE_LEN] + [
        EOS_TOKEN]

    return {
        "en_tokens": en_tokens, "zh_tokens": zh_tokens
    }


training_set = training_set.map(tokenize_example)

print(training_set[0])
