import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
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


def compute_accuracy(model, x_val, y_val, index_to_char, show_sample=False):
    model.eval()
    eq = 0
    tot = 0
    with torch.no_grad():
        for i in range(x_val.shape[0]):
            computed = model.sample(x_val[i])
            v = int("".join([index_to_char[idx] for idx in y_val[i].tolist()][1:]))
            computed = computed.lstrip("0")

            if show_sample and i < 3:
                print('sampled results: ', computed, v)

            try:
                computed_int = int(computed)
                if computed_int == v:
                    eq += 1
            except ValueError:
                pass
            tot += 1

        print('accuracy = ', eq * 1.0 / tot)


def main():
    device = get_device()
    print("using device ", device)

    (x_train, y_train), (x_val, y_val), (char_to_index, index_to_char) = load_data()
    voc_size = len(char_to_index)
    print("voc = ", voc_size)

    model = CalculatorModel(voc_size, char_to_index, index_to_char)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()
    training_loader = DataLoader(AdditionDataset(x_train, y_train), batch_size=128, drop_last=True)

    for epoch in range(50):
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
        compute_accuracy(model, x_val, y_val, index_to_char, True)

        # print(model.sample("99+21"))
        # now for each in validation, sample and try to compare results


main()
