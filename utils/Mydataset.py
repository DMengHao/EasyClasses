import pickle
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import os


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class MyDataset(Dataset):
    def __init__(self, train: bool):
        if train==True:
            with open(os.path.join(ROOT, "data", "train_data_cache.pkl"), 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open(os.path.join(ROOT, "data", "val_data_cache.pkl"), 'rb') as f:
                self.data = pickle.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][1], self.data[idx][0]


if __name__ == '__main__':
    dataset = MyDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        print(i, data)