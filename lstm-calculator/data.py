import numpy as np


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
        q_n = [char_to_index[c] for c in q]
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

    return (x_train, y_train), (x_val, y_val)


(x, y), (x_val, y_val) = load_data()
print(x.shape)
