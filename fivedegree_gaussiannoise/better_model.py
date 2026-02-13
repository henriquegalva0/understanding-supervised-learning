import torch as tt
from torch import nn
import matplotlib.pyplot as plt

from gen_noisy_data import x_values,y_values,x_sample,y_sample, y_clean, dataloader

# - - - - -

mean = x_sample.mean()
std = x_sample.std()

mean_y = y_sample.mean()
std_y = y_sample.std()

# - - - - -

def init_weights(m):
    if isinstance(m, nn.Linear):
        tt.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# - - - - -

class bettermodel(tt.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1= nn.Linear(1,32,bias=True)
        self.linear2= nn.Linear(32,32,bias=True)
        self.linear3= nn.Linear(32,1,bias=True)
        self.Tanh = tt.nn.Tanh()

    def forward(self,x):
        x = (x-mean)/std
        x1 = self.Tanh(self.linear1(x))
        x2 = self.Tanh(self.linear2(x1))
        return self.linear3(x2)

# - - - - -

model = bettermodel()
model.apply(init_weights)

loss_function = nn.MSELoss()
loss_history = list()
learning_rate = 1e-3
epochs = 500

# - - - - -

model.train()

optimizer = tt.optim.SGD(
    model.parameters(),
    lr=learning_rate)

# - - - - -

for epoch in range(epochs):

    epoch_loss = 0

    for batch in dataloader:
        y_normalized = (batch['y'] - mean_y) / std_y
        y_target_prediction = model(batch['x'])
        
        difference = loss_function(y_target_prediction, y_normalized)

        difference.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += difference.item()

    loss_history.append(epoch_loss / len(dataloader))

# - - - - -

plt.plot(range(epochs), loss_history)
plt.xlabel('epoch')
plt.ylabel('Loss (MSE)')
plt.savefig('fivedegree_gaussiannoise/img/loss-epochs.png')