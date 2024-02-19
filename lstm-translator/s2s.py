import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, emb_hidden=100, voc_size=100, lstm_hidden=100):
        super().__init__()
        self.emb_hidden = emb_hidden
        self.voc_size = voc_size

        self.emb = nn.Embedding(voc_size, emb_hidden)
        self.lstm = nn.LSTM(emb_hidden, lstm_hidden, 2, batch_first=True)

    def forward(self, x):
        x = self.emb(x)
        _, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, emb_hidden=100, dest_voc_size=100, lstm_hidden=100):
        super().__init__()

        self.emb = nn.Embedding(dest_voc_size, emb_hidden)
        self.lstm = nn.LSTM(emb_hidden, lstm_hidden, 2, batch_first=True)
        self.affine = nn.Linear(lstm_hidden, dest_voc_size)

    def forward(self, x, h, c):
        x = self.emb(x)
        x, (h, c) = self.lstm(x, (h, c))
        x = self.affine(x)
        return x


class Seq2Seq(nn.Module):
    def __init__(self, src_voc_size, dest_voc_size):
        super().__init__()
        self.encoder = Encoder(voc_size=src_voc_size, emb_hidden=128, lstm_hidden=512)
        self.decoder = Decoder(dest_voc_size=dest_voc_size, emb_hidden=128, lstm_hidden=512)

    def forward(self, source, target):
        h, c = self.encoder(source)

        # target: [seq, batch_size]

        output = self.decoder(target, h, c)
        return output
