from torch import tensor
import torch



tensor_1 = torch.tensor([1., 2., 3.]) 
tensor_2 = torch.tensor([4., 5., 6.]) 
print(tensor_1 + tensor_2) 


# This outputs 

tensor([5., 7., 9.]) 

# We can now use the .to() method. This method is the same as the one we use to change a tensor’s datatype to transfer these tensors onto a GPU and per- form the addition there: 

tensor_1 = tensor_1.to("cuda") 
tensor_2 = tensor_2.to("cuda") 
print(tensor_1 + tensor_2) 

# The output is 

tensor([5., 7., 9.], device='cuda:0') 

# The resulting tensor now includes the device information, device='cuda:0', which means that the tensors reside on the first GPU. If your machine hosts multiple GPUs, you can specify which GPU you’d like to transfer the tensors to. You do so by indicat- ing the device ID in the transfer command. For instance, you can use .to("cuda:0"), .to("cuda:1"), and so on. 

# However, all tensors must be on the same device. Otherwise, the computation will fail,


# Now that we are familiar with transferring tensors to the GPU, we can modify the training loop to run on a GPU. This step requires only changing three lines of code, as shown in the following listing.


torch.manual_seed(123) 
model = NeuralNetwork(num_inputs=2, num_outputs=2) 

device = torch.device("cuda") 
model = model.to(device) 

optimizer = torch.optim.SGD(model.parameters(), lr=0.5) 

num_epochs = 3 

for epoch in range(num_epochs): 
    
    model.train() 
    for batch_idx, (features, labels) in enumerate(train_loader): 
        features, labels = features.to(device), labels.to(device) 
        logits = model(features) 
        loss = F.cross_entropy(logits, labels) # Loss function 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        ### LOGGING 
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}" f" | Batch {batch_idx:03d}/{len(train_loader):03d}" f" | Train/Val Loss: {loss:.2f}") 
        
        model.eval() 
        # Insert optional model evaluation code