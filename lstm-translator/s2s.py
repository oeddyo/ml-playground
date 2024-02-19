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
        return h


class Decoder(nn.Module):
    def __init__(self, emb_hidden=100, voc_size=100, lstm_hidden=100):
        super().__init__()

        self.emb = nn.Embedding(voc_size, emb_hidden)
        self.lstm = nn.LSTM(emb_hidden, lstm_hidden, 2, batch_first=True)

    def forward(self, x, h):
        x = self.emb(x)
        c = torch.zeros_like(h)
        output, (h, c) = self.lstm(x, (h, c))
        return output, (h, c)


class Seq2Seq(nn.Module):
    def __init__(self, src_voc_size, dest_voc_size):
        super().__init__()
        self.encoder = Encoder(voc_size=src_voc_size)
        self.decoder = Decoder(voc_size=dest_voc_size)

    def forward(self, source, target):
        h = self.encoder(source)

        # target: [seq, batch_size]

        output = self.decoder(target, h)
        return output
