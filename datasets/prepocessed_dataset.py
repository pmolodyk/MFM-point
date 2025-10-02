import os
import numpy as np
from torch.utils.data import Dataset
import numpy as np

class PreprocessedPC(Dataset):
    def __init__(self, root_dirs, split='train'):
        paths = [os.path.join(path_, split) for path_ in root_dirs]
        self.path = []
        for path in paths:
            self.path += [os.path.join(path, file) for file in os.listdir(path)]
        self.split = split

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        return {'idx':idx,
                f'{self.split}_points': np.load(self.path[idx], allow_pickle=True)['arr_0'].item()[f'{self.split}_points']}