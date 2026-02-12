# Objective
This repository was created to really understand how to build neural networks and how to make them work properly with matplot, pytorch and numpy.

-----

# 5 Degree Function

## Mission
On the [first example](fivedegree/) of this repository, we're going to build and train a neural network capable of predicting values following the behaviour of a five degree math function.

### Data
Firstly, we'll define an array with _x values_ and _y values_ in [gen_data.py](fivedegree/gen_data.py) to create the target that our neural network must recreate. The following five degree function will be our task:

$$f(x) = x^5 - 6x^3 + 2x $$

![f5dg](fivedegree/img/full_5degree.png)

Now, to train our neural network, it's important to choose a few samples so as it don't get overfitted, neither "easy" to understand. In our example, we'll work with **30 random samples** from **100 total samples**:

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

$$ \mathrm{ReLU}(x) = \max(0, x) $$

Also, the _in & out_ layers of our neural network (2 intermediate features):

$$ y = xA^T + b $$

This linear function will be repeated 3 times for each layer of neurons. It is responsible for calculating the dot product between our _`x` features_ and _weights matrix_ $A^T$, plus a _bias_ _`b`_.

```
class main_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_linear = nn.Linear(1,2,bias=True)
        self.relu = nn.ReLU()
        self.out_linear = nn.Linear(2,1,bias=True)
```

Lastly, inside the class, the feed forward mechanism merges all the previous data/functions by standardization ($Z$-score):

$$z = \frac{x - \mu}{\sigma}$$

on each `x` sample, inserting it in the first layer, and going throw the activation function.

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

This is the magic part of our first example. Here the model will learn how to reproduce our five degree function. The first step is to instantiate our loss function.

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

This function calculates the mean squared error (_MSE, Quadratic Loss or L2 Loss_) returned by our model - when comparing the data predicted to the real data.

```
loss_function = nn.MSELoss()
```

**Disclaimer:** There are many loss functions in Machine Learning; this example is just one interesting approach you can choose.

We will also set the learning rate and the number of epochs that our model will go through.
```
learning_rate = 5e-6
epochs = 10
```

Now, we will not only change our model mode to training mode, but also instantiate **the optimizer**.

```
model.train()

optimizer = tt.optim.SGD(
    model.parameters(),
    lr=learning_rate)
```

The optimizer that I've selected is the _Stochastic Gradient Descent (SGD)_, which will go through all batches of data _"giving directions"_. The picture bellow represents the effects of batch sizes and learning rates to the _SGD_'s behaviour. The arrows are directions given by our loss function which will guide ours model parameters to be more accurate.

![sgdfd](fivedegree/img/sgd_representation.png)
*[Image font](https://www.researchgate.net/figure/This-figure-Shows-multi-SGD-optimizer_fig3_327135988) - This diagram doesn't show exactly how __our optimizer__ is handling our learning rate and batch size. It's merely a representation.*

Lastly, let's create the learning looping that our model will repeat for `n epochs` defined previously.

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
        return {'y':y_val.unsqueeze(dim=-1), 'x':x_val.unsqueeze(dim=-1)} ---> each batch
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

## Changing Parameters

The first thing we must do is: in the architecture of the neural network, inside [build_model.py](fivedegree/build_model.py), there must be a new layer with many more neurons (for example **32 neurons**) with the activation function called between the `in, mid` and `out linears`.
```
class main_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_linear = nn.Linear(1,32,bias=True)
        self.mid_linear = nn.Linear(32,32,bias=True)
        self.out_linear = nn.Linear(32,1,bias=True)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = (x-mean)/std
        x1 = self.relu(self.in_linear(x))
        x2 = self.relu(self.mid_linear(x1))
        return self.out_linear(x2)
```
Then, we will change a few numbers on the optimizer values inside [train_model.py](fivedegree/train_model.py).
```
loss_function = nn.MSELoss()
learning_rate = 1e-3
epochs = 500
```
With a bigger learning rate, the model's parameters will be changed quickly and intensely. Furthermore, it will have more time to learn, since we increased the epochs to 500.

**Disclaimer:** You must keep in mind that while training bigger or smaller models for more or less complex tasks, these values and parameters should always be quite different.

### Evaluation (changing parameters)

After changing completely our _learning rate, epochs_ and _number of neurons_, the results are completely different. As we saw earlier, our optimizer is strongly affected by the architecture, learning rate and batches, so the only work we had to do was turning the _"gradient directions"_ more effective to efficiently change the model's parameters.

![atc5dg](fivedegree/img/after_training_changes.png)

It's clear that the model could replicate most of the five degree function curve, which shows us that it's capable of understanding behaviours based on complex patterns. On the other hand, its size has increased considerably as you may see on diagram bellow...

![nn5dg](fivedegree/img/nn_representation.svg)

_This 4 layers & 32 neurons neural network diagram was developed using the [AlexNail tool](https://alexlenail.me/NN-SVG/)._

## Setup Instructions (First Example)
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

# Gaussian Noise

## Mission
On the [first example](fivedegree/), we solved our task optimizing a neural network to understand the behaviour of a five degree function. Now, the task is quite the same, except for the fact that the data won't be so clean to read.

### Data

Even though the pattern created by selected values from a five degree function is quite complex to understand, if you train with clean data, eventually your model will catch up and easily recognize the pattern.

The task now is to make the learning path of our model harder. We're adding _gaussian noise_ (or white noise) to the data sample in [gen_noisy_data.py](fivedegree_gaussiannoise/gen_noisy_data.py).

$$y_{noisy} = y_{true} + \epsilon$$

The noise factor over our data will be given by random normalized values obtained from a gaussian distribution added to `y_clean`, which is the previous defined `f(x)`, but now with **300 sample values** going from `-3` to `3` to study how the noise affects the model.

$$f(x) = x^5 - 6x^3 + 2x $$

With that, the following code will be responsible for adding noise to the clean data.

```
sigma = 0.5
noise = np.random.normal(0,sigma, y_clean.shape)
y_values = y_clean + noise
```

Ultimately, we'll be selecting only **one tenth** of our full dataset for the training _(we'll be using `target_values = np.random.permutation(300)[:30]` to avoid duplicating sample values)_.

```
x_sample = x_values[target_values]
y_sample = y_values[target_values]
```

![sfdn](fivedegree_gaussiannoise/img/sample_noisy.png)

This plot of our sample data shows how little and noisy the information given to the model is right now. Now, the real mission is to study the behaviour of our model facing this problem:

* Will our model suffer from **overfitting**? In other words, will the model memorize the noise and replicate it, or will it develop a robust regression?

## Model

The archicteture used to build and train our model will be the same as in [build_model.py](fivedegree/build_model.py]) and [train_model.py](fivedegree/train_model.py]), however we'll be running a couple of experiments with the size of the neural network.

Within the file [train_model.py](fivedegree_gaussiannoise/train_model.py), the number of neurons will be changed to 16, 32, and 64 for 2 hidden layers, and then for 3 hidden layers. This manual change is called **Grid Search** and is not the most efficient way to find the ideal neural network architecture, but it clearly shows the difference between each one (great for technical analyses within a research repository).


## Setup Instructions (Second Example)
To run the code, if you haven't done this yet, start by cloning the github repository.
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
Finally, execute the following scripts to run the **second example**.
```
python ./fivedegree_gaussiannoise/gen_noisy_data.py
```

-----

# Additional Notes
- *Some of the graph plotting codes may not be on the code;*
- All the information used to build this repository can be found in the [PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html);
- To avoid excessive information, there **will not be any explanatory comments inside the scripts**. All notes are in [README](README.md);
- Artificial intelligence was not used to build any sort of script in this repository, neither [README](README.md) explanations _(that's why it may contain issues)_;
- If you find any problems in the code/explanations, feel free to reach me out and share your ideas! I am always open to improvements.
