from typing import List, Tuple
import random
import spacy
import torchtext
from spacy import Language
from torchtext.vocab import Vocab

UNKNOWN_TOKEN = "<UNK>"
PADDING_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"

SPECIAL_TOKENS = [UNKNOWN_TOKEN, PADDING_TOKEN, EOS_TOKEN, SOS_TOKEN]

NLP_MAP = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm"
}

LANGUAGE_FILE = {
    "en": "news-commentary-v13.zh-en.en",
    "zh": "news-commentary-v13.zh-en.zh"
}


def read_language_files(src_file: str, dest_file: str) -> List[Tuple[str, str]]:
    """
    read both source language file and dest language file
    :param src_file: source file path
    :param dest_file: dest file path
    :return: list of string pairs
    """
    src_lines = open(src_file).readlines()[:100]
    dest_lines = open(dest_file).readlines()[:100]

    res = []
    for p in zip(src_lines, dest_lines):
        res.append(p)
    return res


def create_voc(nlp: Language, lines: List[str], max_line_length=500) -> Vocab:
    # use nlp to tokenize each line, then create voc
    token_list = []
    for line in lines:
        # truncate sentences that are too long
        line = line[:max_line_length]
        token_list.append([SOS_TOKEN] + [t.text for t in nlp.tokenizer(line)] + [EOS_TOKEN])

    vocab = torchtext.vocab.build_vocab_from_iterator(
        token_list,
        min_freq=1,
        specials=SPECIAL_TOKENS
    )
    return vocab


class TranslationExample:
    def __init__(self, src_text: str, dest_text: str, src_tensor, dest_tensor):
        self.src_text = src_text
        self.dest_text = dest_text
        self.src_tensor = src_tensor
        self.dest_tensor = dest_tensor


def create_tensor(vocab: Vocab, nlp: Language, text: str):

    indices = vocab.lookup_indices([t.text for t in nlp.tokenizer(text)])

    return



class DataImporter:

    def __init__(self, src_lang="en", dest_lang="zh"):
        self.src_lang = src_lang
        self.dest_lang = dest_lang

        src_file = LANGUAGE_FILE[src_lang]
        dest_file = LANGUAGE_FILE[dest_lang]

        src_nlp = spacy.load(NLP_MAP[src_lang])
        dest_nlp = spacy.load(NLP_MAP[dest_lang])

        text_pairs = read_language_files(src_file, dest_file)
        random.shuffle(text_pairs)

        n_pairs = len(text_pairs)
        train_idx = int(0.8 * n_pairs)
        valid_idx = int(0.9 * n_pairs)

        train_text_pairs = text_pairs[:train_idx]

        # build vocabulary only on training set
        self.src_voc = create_voc(src_nlp, [p[0] for p in train_text_pairs])
        self.dest_voc = create_voc(dest_nlp, [p[1] for p in train_text_pairs])

        valid_pairs = text_pairs[train_idx: valid_idx]
        test_pairs = text_pairs[valid_idx:]

        # split

    # read both src and dest files

    # this should take language identifier and
    # 1. split train, test, validate lines
    # 2. build vocabulary, tokenizer on train
    # 3. tokenize both test and validate
    pass


di = DataImporter()
