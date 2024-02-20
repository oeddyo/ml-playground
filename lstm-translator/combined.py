from torchtext.vocab import Vocab
import numpy as np
import tqdm
import spacy
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch
import random

# constants
UNKNOWN_TOKEN = "<UNK>"
PADDING_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"


class TDataset(Dataset):

    def __init__(self, translation_data):
        self.translation_data = translation_data

    def __getitem__(self, item):
        return self.translation_data[item]

    def __len__(self):
        return len(self.translation_data)


def collate_fn(batch):
    batch_src = [e["src_ids"] for e in batch]
    batch_src = nn.utils.rnn.pad_sequence(batch_src, batch_first=True, padding_value=1)

    batch_dest = [e["dest_ids"] for e in batch]
    batch_dest = nn.utils.rnn.pad_sequence(batch_dest, batch_first=True, padding_value=1)
    return batch_src, batch_dest


class TranslationData:
    def __init__(self, min_freq=1, max_sentence_len=200, src_lang="de", dest_lang="en", batch_size=64):
        self.batch_size = batch_size
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
        training_set = load_dataset("bentrevett/multi30k", split="train[:5]", trust_remote_code=True)
        validation_set = load_dataset("bentrevett/multi30k", split="validation[:100]", trust_remote_code=True)

        self.src_nlp = spacy.load("de_core_news_sm")
        self.dest_nlp = spacy.load("en_core_web_sm")

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

    def _get_datasets(self):
        return self.training_set, self.validation_set

    def get_training(self):
        src_voc_size, dest_voc_size = len(self.src_vocab), len(self.dest_vocab)
        training_loader = DataLoader(TDataset(self.training_set), batch_size=self.batch_size, collate_fn=collate_fn)
        return src_voc_size, dest_voc_size, training_loader

    def get_validation(self):
        return DataLoader(TDataset(self.validation_set), batch_size=self.batch_size, collate_fn=collate_fn)

    def get_tensor(self, s):
        tokens = [SOS_TOKEN] + [t.text for t in self.src_nlp.tokenizer(s)] + [EOS_TOKEN]
        indexes = self.src_vocab.lookup_indices(tokens)
        res = torch.LongTensor([indexes])
        return res

    def _tokenize(self, example):
        en_tokens = [t.text for t in self.src_nlp.tokenizer(example[self.src_lang])][
                    :self.max_sentence_len]
        dest_tokens = [t.text for t in self.dest_nlp.tokenizer(example[self.dest_lang])][
                      :self.max_sentence_len]

        # if you return fields here, the map function on Dataset will add them to a new row. If you assign the returned
        # dataset then you have a new dataset with the new fields you want
        return {
            "src_tokens": [SOS_TOKEN] + en_tokens + [EOS_TOKEN],
            "dest_tokens": [SOS_TOKEN] + dest_tokens + [EOS_TOKEN]
        }

    def _vectorize(self, example):
        return {
            "src_ids": self.src_vocab.lookup_indices(example["src_tokens"]),
            "dest_ids": self.dest_vocab.lookup_indices(example["dest_tokens"])
        }


class Encoder(nn.Module):

    def __init__(self, emb_hidden=100, voc_size=100, lstm_hidden=100):
        super().__init__()
        self.emb_hidden = emb_hidden
        self.voc_size = voc_size

        self.emb = nn.Embedding(voc_size, emb_hidden)
        self.lstm = nn.LSTM(emb_hidden, lstm_hidden, 2, batch_first=True, dropout=0.5)

    def forward(self, x):
        x = self.emb(x)
        _, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, emb_hidden=100, dest_voc_size=100, lstm_hidden=100):
        super().__init__()

        self.emb = nn.Embedding(dest_voc_size, emb_hidden)
        self.lstm = nn.LSTM(emb_hidden, lstm_hidden, 2, batch_first=True, dropout=0.5)
        self.affine = nn.Linear(lstm_hidden, dest_voc_size)

    def forward(self, x, h, c):
        x = self.emb(x)
        x, (r_h, r_c) = self.lstm(x, (h, c))
        x = self.affine(x)
        return x, (r_h, r_c)


class Seq2Seq(nn.Module):
    def __init__(self, src_voc_size, dest_voc_size):
        super().__init__()
        self.encoder = Encoder(voc_size=src_voc_size, emb_hidden=256, lstm_hidden=512)
        self.decoder = Decoder(dest_voc_size=dest_voc_size, emb_hidden=256, lstm_hidden=512)
        self.dest_voc_size = dest_voc_size

    def forward(self, source, target, teacher_forcing_ratio=0.5, device=torch.device("cpu")):
        h, c = self.encoder(source)

        batch_size, target_length = target.shape

        # [batch_size, dest_length, target_vocab]
        outputs = torch.zeros((batch_size, target_length, self.dest_voc_size)).to(device)

        input = target[:, 0]
        for i in range(1, target_length):
            input = input.view(-1, 1)

            output, (h, c) = self.decoder(input, h, c)
            # Squeeze the output to remove the sequence length dimension
            output = output.squeeze(1)
            outputs[:, i, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)

            input = target[:, i] if teacher_force else top1

        return outputs


def train_fn(model, data_loader, device):
    model.train()
    losses = []
    for batch in tqdm.tqdm(data_loader):
        src, dest = batch

        src = src.to(device)
        dest = dest.to(device)

        output = model(src, dest[:, :-1], 0.5, device)
        model.zero_grad()

        loss = loss_func(output.transpose(1, 2), dest[:, 1:])
        losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        loss.backward()
        optimizer.step()

    return np.mean(losses)


def validate_fn(model, data_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            src, dest = batch
            src = src.to(device)
            dest = dest.to(device)

            output = model(src, dest[:, :-1], 0, device)
            loss = loss_func(output.transpose(1, 2), dest[:, 1:])
            losses.append(loss.item())
    return np.mean(losses)


def translate_sentence(encoder, decoder, src_tensor, dest_vocab: Vocab, device):
    model.eval()

    with torch.no_grad():
        h, c = encoder(src_tensor)

        inputs = [3]

        for i in range(10):
            x = torch.tensor([inputs]).to(device)
            x, (h, c) = decoder(x, h, c)
            new_idx = x[0, -1].argmax()
            inputs.append(new_idx.item())
            if new_idx == 2:
                break

        result_sent = []
        for idx in inputs[1:]:
            result_sent.append(dest_vocab.get_itos()[idx])
    return result_sent


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == '__main__':
    # configurations
    n_epoch = 20

    device = select_device()

    td = TranslationData(min_freq=2, src_lang="de")

    src_voc_size, dest_voc_size, training_data_loader = td.get_training()
    validation_data_loader = td.get_validation()

    model = Seq2Seq(src_voc_size, dest_voc_size).to(device)

    # 1 is consistent with padding index in data.py
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=1)
    optimizer = torch.optim.Adam(model.parameters(), )
    for epoch in range(n_epoch):
        train_loss = train_fn(model, training_data_loader, device)
        valid_loss = validate_fn(model, validation_data_loader, device)

        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

        input_tensor = td.get_tensor("Ein Mann sieht sich einen Film an").to(device)

        print(translate_sentence(model.encoder, model.decoder, input_tensor, td.dest_vocab, device))
