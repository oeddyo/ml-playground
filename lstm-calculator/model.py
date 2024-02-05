import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, voc_size, num_embeddings=100, num_hidden=100, num_layers=3):
        super().__init__()
        self.emb = nn.Embedding(voc_size, num_embeddings)
        self.lstm = nn.LSTM(num_embeddings, num_hidden, num_layers=num_layers, batch_first=True, dropout=0.5)

    def forward(self, x):
        x = self.emb(x)
        output, (h, c) = self.lstm(x)
        return h


class Decoder(nn.Module):
    def __init__(self, voc_size, num_embeddings=100, num_hidden=100, num_layers=3):
        super().__init__()
        self.emb = nn.Embedding(voc_size, num_embeddings)
        self.lstm = nn.LSTM(num_embeddings, num_hidden, num_layers, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(num_hidden, voc_size, )

    def forward(self, x, h):
        x = self.emb(x)
        output, _ = self.lstm(x, (h, torch.zeros_like(h)))
        output = self.linear(output)
        return output


class CalculatorModel(nn.Module):
    def __init__(self, voc_size, char_to_index, index_to_char):
        super().__init__()
        self.encoder = Encoder(voc_size, num_embeddings=16, num_hidden=128)
        self.decoder = Decoder(voc_size, num_embeddings=16,  num_hidden=128)
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char

    def forward(self, x, t):
        h = self.encoder(x)
        return self.decoder(t, h)

    def sample(self, x_vec):
        x = torch.tensor(x_vec)

        cur = [self.char_to_index["_"]]
        h = self.encoder(x)

        while True:
            t = torch.tensor(cur)
            output = self.decoder(t, h)

            prob = torch.softmax(output, dim=1)[-1]

            #next_char_index = torch.multinomial(prob, num_samples=1).item()
            next_char_index = torch.argmax(prob).item()
            if self.index_to_char[next_char_index] == ' ':
                break
            cur.append(next_char_index)

        cur = [self.index_to_char[i] for i in cur[1:]]
        return "".join(cur)
