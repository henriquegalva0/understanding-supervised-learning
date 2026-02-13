import torch as tt
from torch import nn
import matplotlib.pyplot as plt

from gen_noisy_data import x_values,y_values,x_sample,y_sample, y_clean, dataloader

# - - - - -

mean = x_sample.mean()
std = x_sample.std()

# - - - - -

class bettermodel(tt.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1= nn.Linear(1,32,bias=True)
        self.linear2= nn.Linear(32,32,bias=True)
        self.linear3= nn.Linear(32,1,bias=True)
        self.ReLU = tt.nn.ReLU()

    def forward(self,x):
        x = (x-mean)/std
        x1 = self.ReLU(self.linear1(x))
        x2 = self.ReLU(self.linear2(x1))
        return self.linear3(x2)

# - - - - -

model = bettermodel()

loss_function = nn.MSELoss()
learning_rate = 1e-3
epochs = 500

# - - - - -

model.train()

optimizer = tt.optim.SGD(
    model.parameters(),
    lr=learning_rate)

for epoch in range(epochs):

    for batch in dataloader:
        y_target_prediction = model(batch['x'])
        
        difference = loss_function(y_target_prediction, batch['y'])
        difference.backward()
        
        optimizer.step()
        optimizer.zero_grad()

    print(difference)

