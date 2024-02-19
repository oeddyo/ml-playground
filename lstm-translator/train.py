import s2s
import torch
from data import TranslationData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

td = TranslationData()
src_voc_size, dest_voc_size, training_data_loader = td.get_training()
model = s2s.Seq2Seq(src_voc_size, dest_voc_size).to(device)


def train_fn(model, data_loader):
    model.train()

    for data in data_loader:
        src, dest = data
        print("so src = ", src.shape, dest.shape)

        output = model(src, dest)


train_fn(model, training_data_loader)
