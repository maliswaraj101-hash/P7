# %%
from nnfs.datasets import spiral_data
import numpy as np
import nnfs

x , y = spiral_data(samples=100 , classes=3)


# %%
class Layer_Dense :
    def __init__(self , n_inputs , n_neurons):
        self.weights = 0.01 * np.random.randn(n_neurons , n_inputs)
        self.biases = np.zeros((1 , n_neurons))
    
    def forward(self , inputs):
        self.inputs = inputs
        self.output = np.dot(inputs , self.weights.T) + self.biases

    def backward(self , dvalues):
        self.dweights = np.dot(self.inputs.T , dvalues)
        self.dbiases = np.sum(dvalues , axis= 0 , keepdims=True)

# %%
class Activation_Relu:
    def forward(self , inputs):
        self.inputs = inputs
        self.output = np.maximum(0 , inputs)

    def backward(self , dvalues):
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0 

# %%
class Activation_Softmax:

    def forward(self , inputs):
        exp_values = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))

        probabilities = exp_values /np.sum(exp_values , axis=1 , keepdims = True)
        self.output = probabilities
       

# %%
class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):

    def forward(self , y_pred , y_true):

        if len(y_true.shape) == 1:
            confidences = y_pred[range(len(y_pred)) , y_true]
        else:
            confidences = np.sum(y_pred * y_true , axis = 1)

        return -np.log(confidences)     

    def backward(self , dvalues , y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues

        self.dinputs = self.dinputs/samples
                 

# %%


# %%
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self , inputs , y_true):
        self.activation.forward(inputs)

        self.output = self.activation.output

        return self.loss.calculate(self.output , y_true)
    
    def backward(self , dvalues , y_true):
        samples = len(dvalues)

        if(len(y_true.shape) == 2):
            y_true = np.argmax(y_true , axis=1)
        
        self.dinputs = dvalues.copy()

        self.dinputs[range(samples) , y_true] -= 1

        self.dinputs = self.dinputs / samples

# %%
softmax_outputs = np.array([[0.7 , 0.1 , 0.2],
                            [0.1 , 0.5 , 0.4],
                            [0.02 , 0.9 , 0.08]])

class_targets = np.array([0 , 1 , 1])

softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs , class_targets)
dvalues1 = softmax_loss.dinputs
print("Gradients : combined loss and activation:" )
print(dvalues1)

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# will implement this now

# %%
dense1 = Layer_Dense(2 , 3)

activation1 = Activation_Relu()

dense2 = Layer_Dense(3 , 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output , y)



# %%
print(loss)

# %%
loss_activation.backward(loss_activation.output , y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.inputs)
dense1.backward(activation1.dinputs)

# %%
print(dense1.weights)
print(dense1.biases)
print(dense2.weights)
print(dense2.biases)


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



