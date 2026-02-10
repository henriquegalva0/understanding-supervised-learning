from torch import nn
import torch as tt
import matplotlib.pyplot as plt

from build_model import model
from gen_data import dataloader
from gen_data import x_sample,y_sample,x_values,y_values

# - - - - -

loss_function = nn.MSELoss()
learning_rate = 5e-6
epochs = 10

# - - - - -

model.train()

optimizer = tt.optim.SGD(
    model.parameters(),
    lr=learning_rate)

# - - - - -

for epoch in range(epochs):
    
    for batch in dataloader:
        y_target_prediction = model(batch['x'])
        
        difference = loss_function(y_target_prediction, batch['y'])
        difference.backward()
        
        optimizer.step()
        optimizer.zero_grad()

    print(difference)

# - - - - -

model.eval()

y_apredictions = list()

for x_feature in x_sample:

    with tt.no_grad():
        bprediction = model(tt.tensor([x_feature]))
        treated_bprediction = bprediction.detach().item()
        y_apredictions.append(treated_bprediction)

# - - - - -

plt.plot(x_values,y_values)
plt.plot(x_sample,y_sample,'+',color='red')
plt.plot(x_sample,y_apredictions)
plt.savefig('fivedegree/img/after_training.png')
plt.close()