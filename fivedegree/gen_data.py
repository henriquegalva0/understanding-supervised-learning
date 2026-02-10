import torch as tt
import random as rd
import numpy as np
from torch.utils.data import DataLoader, Dataset

# - - - - -

class main_dataset(Dataset):
    def __init__(self, x_sample, y_sample):
        self.x_sample = x_sample
        self.y_sample = y_sample
    def __len__(self):
        return len(self.y_sample)
    def __getitem__(self, index):
        y_val = tt.tensor(self.y_sample[index])
        x_val = tt.tensor(self.x_sample[index])
        return {'y':y_val.unsqueeze(dim=-1), 'x':x_val.unsqueeze(dim=-1)}
    
# - - - - -

def sample_function(x):
    return x**5 - 6*x**3 + 2*x

# - - - - -

x_values = np.linspace(-2.5, 2.5, 100).astype(np.float32)
y_values = sample_function(x_values)

# - - - - -

target_values = [rd.randint(0, 99) for _ in range(30)]
target_values.sort()

x_sample = x_values[target_values]
y_sample = y_values[target_values]

# - - - - -

data = main_dataset(x_sample,y_sample)
dataloader = DataLoader(data, batch_size=2, shuffle=True)