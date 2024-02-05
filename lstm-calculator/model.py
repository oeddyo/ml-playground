import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, voc_size, num_embeddings=100, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(voc_size, num_embeddings)
        self.lstm = nn.LSTM(num_embeddings, num_embeddings, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x = self.emb(x)
        output, (h, c) = self.lstm(x)
        return h


class Decoder(nn.Module):
    def __init__(self, voc_size, num_embeddings=100, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(voc_size, num_embeddings)
        self.lstm = nn.LSTM(num_embeddings, num_embeddings, num_layers, batch_first=True)
        self.linear = nn.Linear(num_embeddings, voc_size, )

    def forward(self, x, h):
        x = self.emb(x)
        output, _ = self.lstm(x, (h, torch.zeros_like(h)))
        output = self.linear(output)
        return output


class CalculatorModel(nn.Module):
    def __init__(self, voc_size):
        super().__init__()
        self.encoder = Encoder(voc_size)
        self.decoder = Decoder(voc_size)

    def forward(self, x, t):
        h = self.encoder(x)

        return self.decoder(t, h)
