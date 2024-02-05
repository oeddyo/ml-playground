import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from model import CalculatorModel
from data import load_data, AdditionDataset


def get_device():
    # Check for MPS (Apple Silicon GPU) availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def main():
    device = get_device()
    print("using device ", device)

    (x_train, y_train), _, (char_to_index, index_to_char) = load_data()
    voc_size = len(char_to_index)
    print("voc = ", voc_size)

    model = CalculatorModel(voc_size)
    optimizer = SGD(model.parameters(), lr=1)
    criterion = CrossEntropyLoss()
    training_loader = DataLoader(AdditionDataset(x_train, y_train), batch_size=64, drop_last=True)

    for epoch in range(10):
        model.train()
        progress_bar = tqdm(training_loader, desc=f"Epoch {epoch + 1}")

        for in_seq, out_seq in progress_bar:
            optimizer.zero_grad()

            logits = model(in_seq, out_seq[:, :-1])

            loss = criterion(logits.transpose(1, 2), out_seq[:, 1:])
            loss.backward()
            norm = clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            progress_bar.set_postfix({"loss": loss.item(), "norm": norm.item()})


main()
