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

        output = model(src, dest)
        model.zero_grad()

        loss = loss_func(output.transpose(1, 2), dest)
        losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        loss.backward()
        optimizer.step()

    return np.mean(losses)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    td = TranslationData()
    src_voc_size, dest_voc_size, training_data_loader = td.get_training()
    model = s2s.Seq2Seq(src_voc_size, dest_voc_size).to(device)

    # 1 is consistent with padding index in data.py
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=1)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        train_loss = train_fn(model, training_data_loader, device)

        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
