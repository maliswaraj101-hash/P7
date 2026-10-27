import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from multi_NN import NeuralNetwork
from dataloaders import train_loader, test_loader

torch.manual_seed(123)  # Set the random seed for reproducibility
model = NeuralNetwork(num_inputs = 2 , num_outputs = 2)  # Replace with your model class

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # 

num_epochs = 3 # Number of epochs to train

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features , labels ) in enumerate(train_loader):
        logits = model(features)
        # full forward pass

        loss = F.cross_entropy(logits , labels)

        optimizer.zero_grad()
        # Every parameter in your model has a .grad attribute.Before computing new gradients, you clear the old ones.
        loss.backward()
        optimizer.step()

        # The optimizer uses the gradients to update the model parameters

        # If you're using SGD, the update is roughly:
        
        # new_weight = old_weight - learning_rate × gradient
        
        # Now the model's weights have been adjusted based on the current batch.
        
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}" f" | Batch {batch_idx:03d}/{len(train_loader):03d}" f" | Train Loss: {loss:.2f}")

    model.eval()

num_params = sum(param.numel() for param in model.parameters())
print(f"Total number of model parameters: {num_params}")

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable model parameters: {num_trainable_params}")
# params is weights + biases.
# numel() returns the number of elements in a tensor.


#     why optimizer.zero_grad()??
#     If you don't call optimizer.zero_grad(), gradients from previous batches will keep accumulating, leading to incorrect updates.

# Let's see why.

# One weight example

# Suppose your model has just one weight:

# w = 5.0
# Batch 1

# You do:

# loss.backward()

# Suppose the gradient is:

# dw = 2

# PyTorch stores it in

# w.grad

# So now

# w.grad = 2

# Then

# optimizer.step()

# updates the weight.

# Batch 2 (without zero_grad)

# Now you process another batch.

# Suppose its gradient is

# dw = 3

# You might expect

# w.grad = 3

# But that's not what PyTorch does.

# Instead it adds the new gradient:

# Old gradient = 2
# New gradient = 3

# Stored gradient = 5

# So now

# w.grad = 5

# The optimizer updates the weight using 5, not 3.

# Batch 3

# Suppose the gradient is

# dw = -1

# Now

# Old gradient = 5

# New gradient = -1

# Stored gradient = 4

# The gradients keep accumulating.

# What zero_grad() does

# Before every backward pass you write

# optimizer.zero_grad()

# This clears all stored gradients.

# So the training loop becomes

# optimizer.zero_grad()

# loss.backward()

# optimizer.step()

# Now the flow is

# Batch 1
# Clear gradients

# w.grad = 0

# Backward

# w.grad = 2

# Update weight
# Batch 2
# Clear gradients

# w.grad = 0

# Backward

# w.grad = 3

# Update weight
# Batch 3
# Clear gradients

# w.grad = 0

# Backward

# w.grad = -1

# Update weight

# Now each update uses only the current batch's gradient, which is what you usually want.

from dataloaders import X_train , y_train

with torch.no_grad():
    outputs = model(X_train)
print(outputs)

# tensor([[ 2.9320, -4.2563],
#         [ 2.6045, -3.8389],
#         [ 2.1484, -3.2514],
#         [-2.1461,  2.1496],
#         [-2.5004,  2.5210]])

# this is just the forward pass.

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1) 
print(probas)

# tensor([[    0.9992,     0.0008],
#         [    0.9984,     0.0016],
#         [    0.9955,     0.0045],
#         [    0.0134,     0.9866],
#         [    0.0066,     0.9934]])


# outputs is the forward pass value , i.e. the logits. The softmax function converts these logits into probabilities that sum to 1 across the classes for each sample.

# dim 1 means across rows. So for each row, the values will sum to 1.0. This is what we want for class probabilities.

# the first row in the preceding code output. Here, the first value (col- umn) means that the training example has a 99.91% probability of belonging to class 277A.7 A typical training loop 0 and a 0.09% probability of belonging to class 1. (The set_printoptions call is used here to make the outputs more legible.)


# We can convert these values into class label predictions using PyTorch’s argmax function, which returns the index position of the highest value in each row if we set dim=1 (setting dim=0 would return the highest value in each column instead):

predictions = torch.argmax(probas, dim=1)
print(predictions)

# tensor([0, 0, 0, 1, 1])

print(predictions == y_train)
# ofc the training data will be predicted correctly since the model has been trained on it. But this is not a good measure of how well the model will perform on unseen data. We need to evaluate the model on a separate test set to get a better estimate of its performance.

# we got predictions via training data which the model has already learned on labels (y_train).That's why the predictions are correct.



# Using torch.sum, we can count the number of correct predictions: 
print(torch.sum(predictions == y_train))

# --> 5

def compute_accuracy(model, data_loader):
    model = model.eval()  
    correct = 0
    total_examples = 0

    with torch.no_grad():  # Disable gradient computation for inference
        for idx , (features, labels) in enumerate(data_loader):
            
            with torch.no_grad():
                logits = model(features)

            probas = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probas, dim=1)
            compare = predictions == labels
            correct += torch.sum(compare)
            total_examples += len(compare)

    accuracy = correct / total_examples
    return accuracy


print(compute_accuracy(model, test_loader))

# to save the model
torch.save(model.state_dict(), "model.pth")

# The model’s state_dict is a Python dictionary object that maps each layer in the model to its trainable parameters (weights and biases). "model.pth" is an arbitrary filename for the model file saved to disk. We can give it any name and file ending we like; however, .pth and .pt are the most common conventions. Once we saved the model, we can restore it from disk: 

model = NeuralNetwork(2, 2) 
model.load_state_dict(torch.load("model.pth"))

#  the NeuralNetwork(2, 2) architecture needs to match the origi- nal saved model exactly.

#     model = model.eval() puts your neural network into evaluation mode.

# It is usually done before testing or making predictions:

# model.eval()

# with torch.no_grad():
#     outputs = model(X_test)

# Let's understand why.

# Training mode vs Evaluation mode

# A PyTorch model has two modes:

# Training mode (model.train())
# Evaluation mode (model.eval())

# By default, a model starts in training mode.

# print(model.training)

# returns:

# True

# After:

# model.eval()

# it returns:

# False
# Why does a model need different modes?

# Some layers behave differently during training and testing.

# The two important ones are:

# 1. Dropout

# Dropout randomly turns off neurons during training.

# Example:

# Training:

# Neuron outputs:
# [0.5, 0.8, 0.2, 0.9]

# Dropout:

# [0.5, 0, 0.2, 0]

# This helps prevent overfitting.

# During evaluation, you do not want randomness.

# You want all neurons active:

# Testing:

# [0.5, 0.8, 0.2, 0.9]

# model.eval() turns dropout off.

# 2. Batch Normalization

# BatchNorm behaves differently during training and testing.

# During training:

# Use statistics from the current batch:

# Batch mean
# Batch variance

# During evaluation:

# Use running averages collected during training.

# model.eval() switches BatchNorm to use those stored values.

# Important: eval() does NOT stop gradients

# This is a common confusion.

# This:

# model.eval()

# only changes the behavior of certain layers.

# It does not disable gradient calculation.

# For that, you use:

# with torch.no_grad():

# So usually you use both:

# model.eval()

# with torch.no_grad():
#     outputs = model(X_test)

# They do different jobs.

# Training example

# During training:

# model.train()

# for features, labels in train_loader:

#     optimizer.zero_grad()

#     logits = model(features)

#     loss = F.cross_entropy(logits, labels)

#     loss.backward()

#     optimizer.step()

# Here:

# Dropout is active.
# BatchNorm updates its running statistics.
# Gradients are calculated.
# Testing example

# After training:

# model.eval()

# with torch.no_grad():

#     logits = model(X_test)

# Here:

# Dropout is disabled.
# BatchNorm uses learned statistics.
# No gradients are stored.
# About model = model.eval()

# You often see:

# model.eval()

# and:

# model = model.eval()

# Both work.

# Why?

# Because eval() returns the model itself.

# So:

# model.eval()

# does:

# Change model mode
# Return model

# The returned model is the same object.

# Usually people simply write:

# model.eval()

# because assigning it back is unnecessary.

# So the typical workflow is:

# # Training
# model.train()

# # Evaluation
# model.eval()

# with torch.no_grad():
#     predictions = model(X_test)

# In short:

# model.train() → learning mode.
# model.eval() → prediction/testing mode.
# torch.no_grad() → don't calculate gradients.