import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_data():
    lines = open("addition.txt", 'r').readlines()

    questions, answers = [], []
    voc = set()
    for line in lines:
        line = line.replace("\n", "")
        underscore_idx = line.index("_")
        question, answer = line[:underscore_idx], line[underscore_idx:]
        questions.append(question)
        answers.append(answer)
        for c in line:
            voc.add(c)

    sorted_voc = sorted(list(voc))
    char_to_index = {}
    index_to_char = {}
    for i, c in enumerate(sorted_voc):
        char_to_index[c] = i
        index_to_char[i] = c

    xs, ys = [], []
    for q, a in zip(questions, answers):
        # reverse question to make accuracy go up by quite a bit
        q_n = [char_to_index[c] for c in q][::-1]
        a_n = [char_to_index[c] for c in a]
        xs.append(q_n)
        ys.append(a_n)

    indices = np.arange(len(xs))
    np.random.shuffle(indices)

    xs = np.array(xs)
    ys = np.array(ys)

    ratio = 0.9
    train_cutoff = int(ratio * len(indices))

    x_train = xs[:train_cutoff, ]
    y_train = ys[:train_cutoff, ]

    x_val = xs[train_cutoff:, ]
    y_val = ys[train_cutoff:, ]

    return (x_train, y_train), (x_val, y_val), (char_to_index, index_to_char)


class AdditionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


(x_train, y_train), _, (char_to_index, index_to_char) = load_data()
training_loader = DataLoader(AdditionDataset(x_train, y_train), batch_size=64, drop_last=True)

for x, y in training_loader:
    print([index_to_char[i] for i in x[0].tolist()])
    print([index_to_char[i] for i in y[0].tolist()])
    break
