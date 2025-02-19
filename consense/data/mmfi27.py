import torch
import torch.nn as nn
import numpy as np
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
import random
class MMFI27(Dataset):
    
    base_folder = 'mmfi-27-python'

    def __init__(self,train: bool = True,root:str = '') -> None:

        super(MMFI27, self).__init__()

        self.train = train 
        self.data: Any = []
        self.targets = []
        file_name = 'train' if self.train else 'test'
        file_path = os.path.join(root, self.base_folder, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data) # 10,3,114

        self.data = torch.from_numpy(self.data).float()
        self.targets = np.array(self.targets)
        
        print('MMFI27.data',self.data.size())


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        data, target = self.data[index], self.targets[index]

        return data, target

    def __len__(self) -> int:
        return len(self.data)