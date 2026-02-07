# Objective
This repository was created to really understand how to build neural networks and how to make them work properly with matplot, pytorch and numpy.

-----

# 5 Degree Function

## Mission
On the [first example](fivedegree/) of this repository, we're going to build and train a neural network capable of predicting values following the behaviour of a five degree math function.

### Data
Firstly, we'll define an array with _x values_ and _y values_ in [gen_data.py](fivedegree/gen_data.py) to create the target that our neural network must recreate. The following five degree function will be our task:

$$x^5 - 6x^3 + 2x$$

![f5dg](fivedegree/img/full_5degree.png)

Now, to train our neural network, it's important to choose a few samples so as it don't get overfitted. In our example, we'll work with **30 random samples**:

![s5dg](fivedegree/img/sample_5degree.png)

Finally, we must initiate the class of our dataset with the previous samples and load the data using pytorch.

```
class main_dataset(Dataset):
    def __init__(self, x_sample, y_sample):
        self.x_sample = x_sample
        self.y_sample = y_sample
    def __len__(self):
        return len(self.y_sample)
    def __getitem__(self, index):
        return {
            'y':self.y_sample[index].unsqueeze(dim=-1),
            'x':self.x_sample[index].unsqueeze(dim=-1)}

data = main_dataset(x_sample,y_sample)
dataloader = DataLoader(data, batch_size=2, shuffle=True)
```

Our data is ready to be used!

## Model 

Now, with the _x samples_ and _y samples_, we'll develop our model's class. In [build_model.py](fivedegree/build_model.py), we are defining the activation function [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html):

$$
\mathrm{ReLU}(x) = \max(0, x)
$$

Also, the _in & out_ layers of our neural network (2 intermediate features):

```
class main_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_linear = nn.Linear(1,2,bias=True)
        self.relu = nn.ReLU()
        self.out_linear = nn.Linear(2,1,bias=True)
```

Lastly, inside the class, the feedfoward mechanism merges all the previous data/functions by normalizing the x sample, inseting it in the first layer and temporarily going throw the activation function.

```
    def forward(self,x):
        x = (x-mean)/std
        x1 = self.in_linear(x)
        x_temp = self.relu(x1)
        return self.out_linear(x_temp)

model = main_model()
```

Now, our model is almost ready to go. There are still a couple of changes that have to be made. 

-----

# Additional Notes
- All requirements are avaiable on [requirements.txt](requirements.txt);
- To avoid excessive information, there **will not be any comments inside the scripts**. All notes are in [README](README.md).