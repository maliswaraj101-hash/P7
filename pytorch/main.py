import torch 
print(torch.__version__)


# tensors are mathematical objects that can be characterized by their order (or rank), which provides the number of dimen- sions. For example, a scalar (just a number) is a tensor of rank 0, a vector is a tensor of rank 1, and a matrix is a tensor of rank 2,

# From a computational perspective, tensors serve as data containers. For instance, they hold multidimensional data, where each dimension represents a different feature. Tensor libraries like PyTorch can create, manipulate, and compute with these arrays efficiently. In this context, a tensor library functions as an array library.

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1,2,3])

tensor2d = torch.tensor([[1,2,4],[4,5,6]])

tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(tensor0d.dtype)
# torch.int64

floatvec = torch.tensor([1.0, 2.0, 3.0]) 
print(floatvec.dtype)
# torch.float32 

#  A 32-bit floating-point number offers sufficient precision for most deep learning tasks while consuming less memory and computational resources than a 64-bit floating- point number. Moreover, GPU architectures are optimized for 32-bit computations, and using this data type can significantly speed up model training and inference.


# floatvec = tensor1d.to(torch.float32) 
# print(floatvec.dtype) 
# This returns torch.float32 

print(tensor0d.shape)
print(tensor1d.shape)
print(tensor2d.shape)
print(tensor3d.shape)

#  print(tensor2d.shape) -> # torch.Size([2, 3])

# As you can see, .shape returns [2, 3], meaning the tensor has two rows and three col- umns. To reshape the tensor into a 3 × 2 tensor, we can use the .reshape method: 
print(tensor2d.reshape(3, 2))


# Next, we can use .T to transpose a tensor, which means flipping it across its diagonal. Note that this is similar to reshaping a tensor, as you can see based on the follow- ing result:

print(tensor2d.T)

# for matrix multiplication
# can use matmul or @

print(tensor2d @ tensor2d.T)


# # A logistic regression forward pass
# The code in the following listing implements the forward pass (prediction step) of a simple logistic regression classifier, which can be seen as a single-layer neural network. It returns a score between 0 and 1, which is compared to the true class label (0 or 1) when computing the loss.

import torch.nn.functional as F
# target_label_y = torch.tensor([1.0])
# x1 = torch.tensor([1.1])
# w1 = torch.tensor([2.2])

# bias = torch.tensor([0.0])

# z = x1 * w1 + bias

# a = torch.sigmoid(z)

# print(a)

# loss = F.binary_cross_entropy(a, target_label_y)
# print(loss)


# If we carry out computations in PyTorch, it will build a computational graph internally by default if one of its terminal nodes has the requires_grad attribute set to True. This is useful if we want to compute gradients. Gradients are required when training neural networks via the popular backpropagation algorithm, which can be considered an implementation of the chain rule from calculus for neural networks



# PyTorch’s autograd engine constructs a computational graph in the background by tracking every operation per- formed on tensors. Then, calling the grad function, we can compute the gradient of the loss concerning the model parameter w1

from torch.autograd import grad


target_label_y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
# because we want to compute the gradient of the loss with respect to w1, we set requires_grad=True. This tells PyTorch to track operations on this tensor so that it can compute gradients later.
bias = torch.tensor([0.0] , requires_grad=True)

z = x1 * w1 + bias
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, target_label_y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_bias = grad(loss, bias, retain_graph=True)

# By default, PyTorch destroys the computation graph after calculating the gradients to free memory. However, since we will reuse this computation graph shortly, we set retain_graph=True so that it stays in memory. 

print(grad_L_w1)
print(grad_L_bias)

# Here, we have been using the grad function manually, which can be useful for experi- mentation, debugging, and demonstrating concepts. But, in practice, PyTorch pro- vides even more high-level tools to automate this process. For instance, we can call .backward on the loss, and PyTorch will compute the gradients of all the leaf nodes in the graph, which will be stored via the tensors’ .grad attributes:

loss.backward()
print(w1.grad)
print(bias.grad)
    