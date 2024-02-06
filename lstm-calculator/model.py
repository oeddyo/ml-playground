import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, voc_size, num_embeddings, num_hidden, num_layers=3):
        super().__init__()
        self.emb = nn.Embedding(voc_size, num_embeddings)
        self.lstm = nn.LSTM(num_embeddings, num_hidden, num_layers=num_layers, batch_first=True, dropout=0.5)

    def forward(self, x):
        x = self.emb(x)
        output, (h, c) = self.lstm(x)
        return h


class Decoder(nn.Module):
    def __init__(self, voc_size, num_embeddings, num_hidden, num_layers=3):
        super().__init__()
        self.emb = nn.Embedding(voc_size, num_embeddings)
        self.lstm = nn.LSTM(num_embeddings + num_hidden, num_hidden, num_layers, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(num_hidden, voc_size, )

    def forward(self, x, h):
        x = self.emb(x)
        seq_len = x.size(1)

        h_repeat = h[-1].unsqueeze(1).repeat(1, seq_len, 1)  # Shape: [batch, seq_len, num_hidden]
        # Concatenate along the last dimension to include peeky information
        x = torch.cat((x, h_repeat), dim=2)  # New shape: [batch, seq_len, num_embeddings + num_hidden]

        output, _ = self.lstm(x, (h, torch.zeros_like(h)))
        output = self.linear(output)
        return output


class CalculatorModel(nn.Module):
    def __init__(self, voc_size, char_to_index, index_to_char):
        super().__init__()
        self.encoder = Encoder(voc_size, num_embeddings=100, num_hidden=128)
        self.decoder = Decoder(voc_size, num_embeddings=100,  num_hidden=128)
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char

    def forward(self, x, t):
        h = self.encoder(x)
        return self.decoder(t, h)

    def sample(self, x_vec):
        # Ensure x_vec is a tensor with correct dtype and possibly unsqueeze to add batch dimension if necessary
        x = torch.tensor(x_vec, dtype=torch.long).unsqueeze(0)  # Add batch dimension

        cur_idx = [self.char_to_index["_"]]  # Initial character index
        h = self.encoder(x)

        max_cnt = 10
        while max_cnt > 0:
            max_cnt -= 1
            t = torch.tensor([cur_idx], dtype=torch.long)  # Ensure t has batch and sequence dimensions
            output = self.decoder(t, h)

            prob = torch.softmax(output[:, -1, :], dim=1)  # Get probabilities for the last timestep
            #next_char_index = torch.multinomial(prob, num_samples=1).item()
            next_char_index = torch.argmax(prob).item()
            if self.index_to_char[next_char_index] == ' ':
                break
            cur_idx.append(next_char_index)

        cur = [self.index_to_char[i] for i in cur_idx[1:]]
        return "".join(cur)
