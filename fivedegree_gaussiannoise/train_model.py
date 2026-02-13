import torch as tt
from torch import nn
import matplotlib.pyplot as plt

from gen_noisy_data import x_values,y_values,x_sample,y_sample, y_clean, dataloader

# - - - - -

mean = x_sample.mean()
std = x_sample.std()

# - - - - -

class two_layer_model(tt.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.linear1= nn.Linear(1,n,bias=True)
        self.linear2= nn.Linear(n,n,bias=True)
        self.linear3= nn.Linear(n,1,bias=True)
        self.ReLU = tt.nn.ReLU()

    def forward(self,x):
        x = (x-mean)/std
        x1 = self.ReLU(self.linear1(x))
        x2 = self.ReLU(self.linear2(x1))
        return self.linear3(x2)

class three_layer_model(tt.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.linear1= nn.Linear(1,n,bias=True)
        self.linear2= nn.Linear(n,n,bias=True)
        self.linear3= nn.Linear(n,n,bias=True)
        self.linear4= nn.Linear(n,1,bias=True)
        self.ReLU = tt.nn.ReLU()

    def forward(self,x):
        x = (x-mean)/std
        x1 = self.ReLU(self.linear1(x))
        x2 = self.ReLU(self.linear2(x1))
        x3 = self.ReLU(self.linear3(x2))
        return self.linear4(x3)
    
# - - - - -

def optimizing(selected_model):

    selected_model.train()

    optimizer = tt.optim.SGD(
        selected_model.parameters(),
        lr=learning_rate)

    for epoch in range(epochs):

        for batch in dataloader:
            y_target_prediction = selected_model(batch['x'])
            
            difference = loss_function(y_target_prediction, batch['y'])
            difference.backward()
            
            optimizer.step()
            optimizer.zero_grad()

        print(f'model: {selected_model}; difference:',difference)

def evaluating(selected_model):

    selected_model.eval()

    y_predictions = list()

    for x_feature in x_sample:

        with tt.no_grad():
            prediction = selected_model(tt.tensor([x_feature]))
            treated_prediction = prediction.detach().item()
            y_predictions.append(treated_prediction)

    return y_predictions

# - - - - -

loss_function = nn.MSELoss()
learning_rate = 1e-3
epochs = 500

# - - - - -

twolayer_32 = two_layer_model(n=32)
twolayer_64 = two_layer_model(n=64)
twolayer_128 = two_layer_model(n=128)

three_layer_model_32 = three_layer_model(n=32)
three_layer_model_64 = three_layer_model(n=64)
three_layer_model_128 = three_layer_model(n=128)

# - - - - -

optimizing(twolayer_32)
optimizing(twolayer_64)
optimizing(twolayer_128)

optimizing(three_layer_model_32)
optimizing(three_layer_model_64)
optimizing(three_layer_model_128)

# - - - - -

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes_flat = axes.flatten()

predictions = [
    evaluating(twolayer_32), 
    evaluating(twolayer_64), 
    evaluating(twolayer_128), 
    evaluating(three_layer_model_32), 
    evaluating(three_layer_model_64),
    evaluating(three_layer_model_128)]

titles = ["2layer 32", "2layer 64", "2layer 128", "3layer 32", "3layer 64", "3layer 128"]

for idx in range(6):
    ax = axes_flat[idx]
    
    ax.plot(x_values, y_values)
    ax.plot(x_sample, y_sample, '+', color='red')
    ax.plot(x_sample, predictions[idx])
    
    ax.set_title(titles[idx])
    ax.legend()

plt.tight_layout()
plt.savefig('fivedegree_gaussiannoise/img/training_grid.png')
plt.close()