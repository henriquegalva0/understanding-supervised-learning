from torch import nn
import torch as tt
import matplotlib.pyplot as plt

from gen_data import x_sample,y_sample,x_values,y_values

# - - - - -

mean = x_sample.mean()
std = x_sample.std()

# - - - - -

class main_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_linear = nn.Linear(1,2,bias=True)
        self.relu = nn.ReLU()
        self.out_linear = nn.Linear(2,1,bias=True)
    
    def forward(self,x):
        x = (x-mean)/std
        x1 = self.in_linear(x)
        x_temp = self.relu(x1)
        return self.out_linear(x_temp)

# - - - - -

model = main_model()

# - - - - -

model.eval()

y_bpredictions = list()

for x_feature in x_sample:

    with tt.no_grad():
        bprediction = model(tt.tensor([x_feature]))
        treated_bprediction = bprediction.detach().item()
        y_bpredictions.append(treated_bprediction)

# - - - - -

plt.plot(x_values,y_values)
plt.plot(x_sample,y_sample,'+',color='red')
plt.plot(x_sample,y_bpredictions)
plt.savefig('fivedegree/img/before_training.png')
plt.close()