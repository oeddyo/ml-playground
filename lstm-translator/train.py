import s2s
import torch
from data import TranslationData


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


td = TranslationData()

src_voc_size, dest_voc_size, training = td.get_training()

model = s2s.Seq2Seq(src_voc_size, dest_voc_size).to(device)


for epoch in range(10):

    




