import torch
import torch.nn as nn

class NeuralNetwork(torch.nn.Module):
    def __init__(self , num_inputs , num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            
            # 1st hidden layer
            torch.nn.Linear(num_inputs , 30),
            torch.nn.ReLU(),
            # The Linear layer takes the number of input and output nodes as arguments

            # 2nd hidden layer
            torch.nn.Linear(30 , 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20 , num_outputs)
        )

    
    def forward(self , x):
        logits = self.layers(x)
        return logits


model = NeuralNetwork(num_inputs=50 , num_outputs=3)

# print(model)
# # print(list(model.parameters()))

# cnt = 0
# for p in list(model.parameters()):
#     if p.requires_grad:
#         cnt += 1

# print(cnt)


# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
# print("Total number of trainable model parameters:", num_params)         

print(model.layers[0].weight)

torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
print(out)


# The parameter x in:

# def forward(self, x):
#     logits = self.layers(x)
#     return logits
# represents the input data to the network. More specifically, it contains the input features (often for a whole batch of samples).

# Example 1: One sample
# Suppose each sample has 50 features.

# model = NeuralNetwork(50, 3)

# x = torch.randn(50)
# Here:

# x.shape = (50,)
# x is a single feature vector:

# [0.12, -1.5, 0.87, ..., 2.1]   ← 50 numbers
# When you do:

# output = model(x)
# the forward() method receives:

# x = tensor([...50 values...])
# Example 2: Batch of 64 samples (most common)
# x = torch.randn(64, 50)
# Now:

# x.shape = (64, 50)
# Meaning:

# 64 samples

# Each sample has 50 features

#           Feature1  Feature2 ... Feature50
# Sample1      ...       ...          ...
# Sample2      ...       ...          ...
# ...
# Sample64     ...       ...          ...
# Now inside forward():

# def forward(self, x):
#     print(x.shape)
# prints:

# torch.Size([64, 50])
# Then the first layer
# torch.nn.Linear(50, 30)
# takes each row (50 features) and converts it to 30 outputs.

# So:

# Input x
# (64, 50)
#     │
#     ▼
# Linear(50 → 30)
#     │
#     ▼
# (64, 30)
# Then:

# (64, 30)
#     │
#     ▼
# Linear(30 → 20)
#     │
#     ▼
# (64, 20)
# Then:

# (64, 20)
#     │
#     ▼
# Linear(20 → 3)
#     │
#     ▼
# (64, 3)

# print(out) gives the output -> tensor([[-0.0188,  0.1898,  0.1117]], grad_fn=<AddmmBackward0>)
# means This tensor was produced by a matrix multiplication followed by an addition.
# as output = input × weightᵀ + bias



#  when we use a model for inference (for instance, making predictions) rather than training, the best practice is to use the torch.no_grad() con- text manager. This tells PyTorch that it doesn’t need to keep track of the gradients, which can result in significant savings in memory and computation:

# with torch.no_grad(): 
#    out = model(X) 
# print(out)

# The result is 
# tensor([[-0.1262, 0.1080, -0.1792]])


# PyTorch’s commonly used loss functions combine the softmax (or sigmoid for binary classification) operation with the negative log-likelihood loss in a single class. The reason for this is numerical efficiency and stability. So, if we want to com- pute class-membership probabilities for our predictions, we have to call the softmax function explicitly: 
# with torch.no_grad(): 
#     out = torch.softmax(model(X), dim=1) 
# print(out) 
# This prints tensor([[0.3113, 0.3934, 0.2952]])) 
# The values can now be interpreted as class-membership probabilities that sum up to 1. The values are roughly equal for this random input, which is expected for a randomly initialized model without training.

# it means that during training , the logits are passed to the loss function (like CrossEntropyLoss) which internally applies softmax and computes the loss in one single operation. However, during inference, if you want to interpret the outputs as probabilities, you need to apply softmax explicitly.