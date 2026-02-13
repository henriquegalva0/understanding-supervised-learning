import torch as tt
from torch import nn
import matplotlib.pyplot as plt

from gen_noisy_data import x_values,y_values,x_sample,y_sample, y_clean, dataloader

# - - - - -

mean = x_sample.mean()
std = x_sample.std()

mean_y = y_sample.mean()
std_y = y_sample.std()

learning_rate = 1e-3
loss_function = nn.MSELoss()

wd1=1e-3
wd2=1e-4
wd3=1e-5

# - - - - -

def init_weights(m):
    if isinstance(m, nn.Linear):
        tt.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# - - - - -

class bettermodel(tt.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1= nn.Linear(1,16,bias=True)
        self.linear2= nn.Linear(16,16,bias=True)
        self.linear3= nn.Linear(16,1,bias=True)
        self.Tanh = tt.nn.Tanh()

    def forward(self,x):
        x = (x-mean)/std
        x1 = self.Tanh(self.linear1(x))
        x2 = self.Tanh(self.linear2(x1))
        return self.linear3(x2)

# - - - - -

def train_eval(wd,ep,selected_model):

    selected_model.train()

    optimizer = tt.optim.Adam(
        selected_model.parameters(),
        lr=learning_rate,
        weight_decay=wd)
    
    for _ in range(ep):

        for batch in dataloader:
            y_normalized = (batch['y'] - mean_y) / std_y
            y_target_prediction = selected_model(batch['x'])
            
            difference = loss_function(y_target_prediction, y_normalized)

            difference.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(difference)
    print(f'\n\n\nwd:{wd} ; ep:{ep}\n\n\n')

    selected_model.eval()

    y_predictions = list()

    for x_feature in x_sample:
        with tt.no_grad():
            prediction = selected_model(tt.tensor([x_feature]).float())
            denormalized_prediction = (prediction.item() * std_y) + mean_y
            y_predictions.append(denormalized_prediction)

    return y_predictions

# - - - - -

model_5_3 = bettermodel()
model_5_3.apply(init_weights)
model_5_4 = bettermodel()
model_5_4.apply(init_weights)
model_5_5 = bettermodel()
model_5_5.apply(init_weights)


model_10_3 = bettermodel()
model_10_3.apply(init_weights)
model_10_4 = bettermodel()
model_10_4.apply(init_weights)
model_10_5 = bettermodel()
model_10_5.apply(init_weights)

# - - - - -

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes_flat = axes.flatten()

predictions = [
    train_eval(wd=wd1,ep=500,selected_model=model_5_3), 
    train_eval(wd=wd2,ep=500,selected_model=model_5_4),
    train_eval(wd=wd3,ep=500,selected_model=model_5_5),

    train_eval(wd=wd1,ep=1000,selected_model=model_10_3),
    train_eval(wd=wd2,ep=1000,selected_model=model_10_4),
    train_eval(wd=wd3,ep=1000,selected_model=model_10_5)]

titles = ["500e-1e-3wd", "500e-1e-4wd", "500e-1e-5wd", "1000e-1e-3wd", "1000e-1e-4wd", "1000e-1e-5wd"]

for idx in range(6):
    ax = axes_flat[idx]
    
    ax.plot(x_values, y_values, color='yellow')
    ax.plot(x_values, y_clean, color='blue')
    ax.plot(x_sample, y_sample, '+', color='red')
    ax.plot(x_sample, predictions[idx], color='orange')
    
    ax.set_title(titles[idx])
    ax.legend()

plt.tight_layout()
plt.savefig('fivedegree_gaussiannoise/results/training_grid_16neurons.png')
plt.close()