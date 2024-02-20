import torch
import random
import torch.nn as nn


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

            input = target[:, i] #if teacher_force else top1

        return outputs
