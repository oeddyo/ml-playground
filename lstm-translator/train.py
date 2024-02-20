from torchtext.vocab import Vocab

import s2s
import torch
from data import TranslationData
import numpy as np


def train_fn(model, data_loader, device):
    model.train()
    losses = []
    for batch in data_loader:
        src, dest = batch

        src = src.to(device)
        dest = dest.to(device)

        output = model(src, dest[:, :-1], 0.75, device)
        model.zero_grad()

        loss = loss_func(output.transpose(1, 2), dest[:, 1:])
        losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        loss.backward()
        optimizer.step()

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
    device = select_device()

    td = TranslationData()
    src_voc_size, dest_voc_size, training_data_loader = td.get_training()
    model = s2s.Seq2Seq(src_voc_size, dest_voc_size).to(device)

    # 1 is consistent with padding index in data.py
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1000):
        train_loss = train_fn(model, training_data_loader, device)

        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")

        input_tensor = td.get_tensor("The tendency is either excessive restraint (Europe) or a diffusion of the effort (the United States).").to(device)

        print(translate_sentence(model.encoder, model.decoder, input_tensor, td.dest_vocab, device))
