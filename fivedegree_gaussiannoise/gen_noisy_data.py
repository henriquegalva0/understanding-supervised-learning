import torch as tt
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

# - - - - -

class noisy_dataset(Dataset):
    def __init__(self,x_sample,y_values):
        self.x_sample = x_sample
        self.y_sample = y_sample
    def __len__(self):
        return len(self.y_sample)
    def __getitem__(self, index):
        x_tensor = tt.tensor(self.x_sample[index],dtype=tt.float32)
        y_tensor = tt.tensor(self.y_sample[index],dtype=tt.float32)
        return {'x':x_tensor.unsqueeze(dim=-1),'y':y_tensor.unsqueeze(dim=-1)}

# - - - - -

def sample_function(x):
    return x**5 - 6*x**3 + 2*x

# - - - - -

x_values = np.linspace(-3, 3, 300).astype(np.float32)
y_clean = sample_function(x_values)

# - - - - -

sigma = y_clean.std() * 0.2
noise = np.random.normal(0, sigma, size=y_clean.shape).astype(np.float32)
y_values = y_clean + noise

# - - - - -

target_values = np.random.permutation(300)[:30]
target_values.sort()

x_sample = x_values[target_values]
y_sample = y_values[target_values]

# - - - - -

data = noisy_dataset(x_sample,y_sample)
dataloader = DataLoader(data, batch_size=2, shuffle=True)

# - - - - -

plt.plot(x_sample,y_sample,'+')
plt.savefig('fivedegree_gaussiannoise/img/sample_betternoisy.png')
plt.close()