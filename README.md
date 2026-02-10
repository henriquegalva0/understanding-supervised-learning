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

Finally, we must initiate the class of our dataset with the previous samples (now as tensors of 1 dimension) and load the data using pytorch.

```
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

Although our model is ready to go, there're still a couple of changes that have to be made.

### Evaluation (before training)

For educational purposes, we'll be putting our model to the test before the training. Firstly, let's change it's state to an evaluation mode, so as the inference data doesn't conflict with any sort of training data and create a list for the untrained model predictions.
```
model.eval()
y_bpredictions = list()
```
Now, the inference script comes in action. For each _x value_ selected from the _x sample values_, the model will, **without gradient (since we don't want to calculate the gradients for retropropagation)**, predict a target _y_ based on a _x_ feature; treat the tensor predicted by creating a new tensor and extracting it's value; appending it to the previous untrained model predictions list.
```
for x_feature in x_sample:

    with tt.no_grad():
        bprediction = model(tt.tensor([x_feature]))
        treated_bprediction = bprediction.detach().item()
        y_bpredictions.append(treated_bprediction)
```
As you see on the graph below: the orange line represents the results that, of course, aren't good, since the model is completely clueless of what is happening...

![bt5dg](fivedegree/img/before_training.png)

Now, our task is to train and optimize the model's parameters until it is capable of predicting the behaviour of the five degree function.

## Training

This is the magic part of our first example. Here the model will learn how to reproduce our five degree function.

The first step is to instantiate our loss function (it acts like _"an AI model's teacher"_), set the learning rate and the number of epochs our model will go through.
```
loss_function = nn.MSELoss()
learning_rate = 5e-6
epochs = 10
```
Now, we will not only change our model mode to training mode, but also instantiate the optimizer (responsible for making the values obtained from the loss function useful for the model to learn).
```
model.train()

optimizer = tt.optim.SGD(
    model.parameters(),
    lr=learning_rate)
```
Lastly, let's create the learning looping that our model will repeat for _n epochs_.
```
for epoch in range(epochs):
    
    for batch in dataloader:
        y_target_prediction = model(batch['x'])
        
        difference = loss_function(y_target_prediction, batch['y'])
        difference.backward()
        
        optimizer.step()
        optimizer.zero_grad()

    print(difference)
```
In details:
- **for batch in dataloader:**
When we previously defined our `__getitem__ ` function, the dictionary returned will be loaded by the model as a `batch`.
```
    def __getitem__(self, index):
        y_val = tt.tensor(self.y_sample[index])
        x_val = tt.tensor(self.x_sample[index])
        return {'y':y_val.unsqueeze(dim=-1), 'x':x_val.unsqueeze(dim=-1)}
```

- **y_target_prediction = model(batch['x'])**
Here, as we defined our `foward` function, the `batch['x']` values will be used to predict a `y_target_prediction` by executing the _"in linear function"_, the _ReLU function_ and the _"out linear function"_ over the _"x"_ feature.
```
    def forward(self,x):
        x = (x-mean)/std
        x1 = self.in_linear(x)
        x_temp = self.relu(x1)
        return self.out_linear(x_temp)
```

- **difference = loss_function(y_target_prediction, batch['y'])**
That's the function responsible for telling our model how wrong it is, `y_target_prediction`, from the real target value `batch['y']`.

- **difference.backward()**
Most known as *retropropagation*, this function calculates all the lost function gradients in relation to the trainable weights and bias of the model, then indicates how much each parameter must be ajusted for the model to have more success.

- **optimizer.step()**
The gradients obtained previously by `difference.backward()` will be applied to the model and all parameters should be updated.

- **optimizer.zero_grad()**
Because we're going through multiple batches, we can't sum all of the gradients every loop, so we turn them all to zero after updating the parameters.

- **print(difference)**
Using this line, you can clearly see the change made by the optimizer on the tensors. The terminal should show something close to the following feedback.
```
    tensor(60.3679, grad_fn=<MseLossBackward0>)
    tensor(51.5132, grad_fn=<MseLossBackward0>)
    tensor(63.8576, grad_fn=<MseLossBackward0>)
    tensor(20.4159, grad_fn=<MseLossBackward0>)
    tensor(31.5784, grad_fn=<MseLossBackward0>)
    tensor(46.8480, grad_fn=<MseLossBackward0>)
    tensor(1.9894, grad_fn=<MseLossBackward0>)
    tensor(1.9903, grad_fn=<MseLossBackward0>)
    tensor(44.5663, grad_fn=<MseLossBackward0>)
    tensor(1.9711, grad_fn=<MseLossBackward0>)
```

### Evaluation (after training)
The moment of truth has come. Now we are going to change the mode of the model to evaluation to see how did it went predicting the behaviour of the five degree function after training (the same way we did before training).
```
model.eval()

y_apredictions = list()

for x_feature in x_sample:

    with tt.no_grad():
        bprediction = model(tt.tensor([x_feature]))
        treated_bprediction = bprediction.detach().item()
        y_apredictions.append(treated_bprediction)
```
And the results... didn't change?

![at5dg](fivedegree/img/after_training.png)

That's because a five degree function has such a complexity that our little neural network can't stand a chance. Let's change a little bit our parameters to see what happens.

## Setup Instructions
To run the code and test the neural network model, start by cloning the github repository.
```
git init
git clone https://github.com/henriquegalva0/understanding-supervised-learning.git
```
Create a python environment, activate it and install all project requirements.
```
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```
Finally, execute the following scripts.
```
python ./fivedegree/gen_data.py
python ./fivedegree/build_model.py
python ./fivedegree/train_model.py
```

-----

# Additional Notes
- *Some of the graph plotting codes may not be on the code;*
- To avoid excessive information, there **will not be any explanatory comments inside the scripts**. All notes are in [README](README.md).