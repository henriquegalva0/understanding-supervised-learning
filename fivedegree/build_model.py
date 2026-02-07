from torch import nn

from gen_data import x_sample,y_sample

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