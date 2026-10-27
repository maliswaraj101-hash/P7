import torch

# five training examples with two features each. Accompanying the training examples, we also create a tensor containing the corre- sponding class labels: three examples belong to class 0, and two examples belong to class 1

X_train = torch.tensor([
    [-1.2, 3.1], 
    [-0.9, 2.9], 
    [-0.5, 2.6], 
    [2.3, -1.1], 
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])


X_test = torch.tensor([
    [-0.8, 2.8], 
    [2.6, -1.6]
])
y_test = torch.tensor([0, 1])



# PyTorch requires that class labels start with label 0, and the largest class label value should not exceed the number of output nodes minus 1 (since Python index counting starts at zero). So, if we have class labels 0, 1, 2, 3, and 4, the neural network output layer should consist of five nodes.


from torch.utils.data import Dataset  , DataLoader

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, idx):
        one_x = self.features[idx]
        one_y = self.labels[idx]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]
    

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)


# Now that we’ve defined a PyTorch Dataset class we can use for our toy dataset, we can use PyTorch’s DataLoader class to sample from it


torch.manual_seed(123)
# will shuffle in exactly the same way every time you run the program.

train_loader = DataLoader(
    dataset = train_ds,  
    # The ToyDataset instance created earlier serves as input to the data loader.
    batch_size = 2,
    shuffle = True,
    # Whether or not to shuffle the data
    num_workers = 0
    # The number of background processes
)


test_loader = DataLoader( 
    dataset=test_ds, 
    batch_size=2, 
    shuffle=False, 
    num_workers=0
)


# What shuffle=True does

# At the beginning of each epoch, the DataLoader generates a new random permutation of the dataset.

# For example, suppose your dataset is:

# 1 2 3 4 5 6

# With shuffle=False, every epoch sees:

# Epoch 1: 1 2 3 4 5 6
# Epoch 2: 1 2 3 4 5 6
# Epoch 3: 1 2 3 4 5 6

# With shuffle=True, it might be:

# Epoch 1: 5 1 4 2 6 3
# Epoch 2: 2 6 1 5 3 4
# Epoch 3: 3 4 6 1 5 2

# Notice that the order changes every epoch.

# Then what does torch.manual_seed(123) do?

# The seed initializes PyTorch's random number generator.

# Without a seed:

# Run 1

# Epoch 1: 5 1 4 2 6 3
# Epoch 2: 2 6 1 5 3 4

# Run 2

# Epoch 1: 3 6 2 5 1 4
# Epoch 2: 1 4 6 2 5 3

# The results differ between runs.

# With

# torch.manual_seed(123)

# you get the same sequence of shuffles every time you rerun the program.

# For example:

# Run 1

# Epoch 1: 5 1 4 2 6 3
# Epoch 2: 2 6 1 5 3 4
# Epoch 3: 3 4 6 1 5 2

# Run 2

# Epoch 1: 5 1 4 2 6 3
# Epoch 2: 2 6 1 5 3 4
# Epoch 3: 3 4 6 1 5 2

# So the entire sequence of random shuffles is reproducible, but each epoch still gets a different shuffle.

# Why shuffle at all?

# Imagine your data is sorted by class:

# Sample 1  Class 0
# Sample 2  Class 0
# Sample 3  Class 0
# Sample 4  Class 1
# Sample 5  Class 1
# Sample 6  Class 1

# With batch_size=2 and shuffle=False, the batches are:

# Batch 1: Class 0, Class 0
# Batch 2: Class 0, Class 1
# Batch 3: Class 1, Class 1

# The model sees long stretches of one class, which can make training less effective.

# With shuffle=True, the batches might be:

# Batch 1: Class 1, Class 0
# Batch 2: Class 0, Class 1
# Batch 3: Class 1, Class 0

# Each mini-batch is more representative of the whole dataset, which usually leads to more stable and effective learning.

# In summary
# shuffle=True: changes the sample order every epoch to improve training.
# torch.manual_seed(123): makes those random shuffles reproducible across separate runs of your program.
# These are complementary: one provides randomness for learning, the other provides reproducibility for debugging and experiments.


# During training, the model processes one batch at a time, not the entire dataset at once.

# Let's walk through your example.

# Suppose your original dataset is:

# Sample    Features             Label
# -------------------------------------
# 0         [-1.2,  3.1]          0
# 1         [-0.5,  2.6]          0
# 2         [ 2.3, -1.1]          1
# 3         [-0.9,  2.9]          0
# 4         [ 2.7, -1.5]          1

# Assume:

# batch_size = 2
# shuffle = True
# Step 1: Beginning of an epoch

# Before training starts, DataLoader shuffles the indices.

# Suppose it generates this order:

# 0, 1, 2, 3, 4

# (Another epoch might generate 3, 0, 4, 1, 2.)

# Step 2: Create batches

# Since batch_size=2, it groups the shuffled samples:

# Batch 1:
# Sample 0
# Sample 1

# Batch 2:
# Sample 2
# Sample 3

# Batch 3:
# Sample 4

# which becomes

# Batch 1
# Features:
# [[-1.2, 3.1],
#  [-0.5, 2.6]]

# Labels:
# [0, 0]
# Batch 2
# Features:
# [[ 2.3, -1.1],
#  [-0.9,  2.9]]

# Labels:
# [1, 0]
# Batch 3
# Features:
# [[ 2.7, -1.5]]

# Labels:
# [1]

# The last batch has only one sample because 5 isn't divisible by 2.

# Step 3: Feed Batch 1 into the model

# Your training loop looks like this:

# for features, labels in train_loader:

#     predictions = model(features)

#     loss = loss_fn(predictions, labels)

#     optimizer.zero_grad()

#     loss.backward()

#     optimizer.step()
# First iteration

# features is

# tensor([
#     [-1.2, 3.1],
#     [-0.5, 2.6]
# ])

# labels is

# tensor([0, 0])

# The model processes both samples together.

# 2 samples
#      │
#      ▼
#  Neural Network
#      │
#      ▼
# Predictions

# Example output:

# tensor([
#     [0.90, 0.10],
#     [0.85, 0.15]
# ])

# Loss is computed using the predictions and labels:

# Predictions
#       │
#       ▼
# Loss Function
#       ▲
#       │
# Labels

# Then

# loss.backward()

# computes gradients from both samples in the batch, and

# optimizer.step()

# updates the weights once.

# Step 4: Move to Batch 2

# Now the model receives

# tensor([
#     [ 2.3, -1.1],
#     [-0.9,  2.9]
# ])

# with labels

# tensor([1, 0])

# Again:

# Batch 2
#       │
#       ▼
# Forward Pass
#       │
#       ▼
# Loss
#       │
#       ▼
# Backward Pass
#       │
#       ▼
# Weight Update
# Step 5: Batch 3

# Finally,

# tensor([[2.7, -1.5]])

# with

# tensor([1])

# goes through the exact same process.

# End of Epoch

# After Batch 3, the model has seen every sample exactly once.

# Epoch 1

# Batch 1 ✓
# Batch 2 ✓
# Batch 3 ✓

# All training samples processed.
# Beginning of Epoch 2

# Because shuffle=True, the DataLoader creates a new random order.

# For example:

# 3, 0, 4, 1, 2

# Now the batches become

# Batch 1
# Sample 3
# Sample 0

# Batch 2
# Sample 4
# Sample 1

# Batch 3
# Sample 2

# Notice that the samples are the same, but their order and grouping into batches has changed.

# Complete Flow
# Training Dataset
#        │
#        ▼
# DataLoader
# (shuffle=True)
#        │
#        ▼
# Random order of samples
#        │
#        ▼
# Split into batches
#        │
#        ▼
# Batch 1 ─► Forward Pass ─► Loss ─► Backward Pass ─► Update Weights
#        │
#        ▼
# Batch 2 ─► Forward Pass ─► Loss ─► Backward Pass ─► Update Weights
#        │
#        ▼
# Batch 3 ─► Forward Pass ─► Loss ─► Backward Pass ─► Update Weights
#        │
#        ▼
# End of Epoch
#        │
#        ▼
# Shuffle Again
#        │
#        ▼
# Next Epoch

# A key point is that the weights are updated after each batch, not after each individual sample and not only after the entire dataset. This is why this approach is called mini-batch gradient descent.


# That's the right way to think about it.

# Suppose you have 5 samples:

# A B C D E

# You set:

# torch.manual_seed(123)

# train_loader = DataLoader(
#     train_ds,
#     batch_size=2,
#     shuffle=True
# )

# and train for 10 epochs.

# Run 1

# The DataLoader might produce these shuffles:

# Epoch 1: C A E B D
# Epoch 2: B E A D C
# Epoch 3: D C B A E
# Epoch 4: A D C E B
# ...
# Epoch10: E B D A C

# Notice that every epoch has a different order.

# Run 2 (same code, same seed)

# You restart the entire program and run it again.

# Because you used the same seed, you'll get exactly the same sequence:

# Epoch 1: C A E B D
# Epoch 2: B E A D C
# Epoch 3: D C B A E
# Epoch 4: A D C E B
# ...
# Epoch10: E B D A C

# So:

# Epoch 1 in Run 1 = Epoch 1 in Run 2 ✅
# Epoch 2 in Run 1 = Epoch 2 in Run 2 ✅
# ...
# Epoch 10 in Run 1 = Epoch 10 in Run 2 ✅

# The randomness is reproducible.

# If you don't set the seed

# Then the two runs might look like this:

# Run 1

# Epoch 1: C A E B D
# Epoch 2: B E A D C
# Epoch 3: D C B A E

# Run 2

# Epoch 1: A D C B E
# Epoch 2: E C A D B
# Epoch 3: B D E A C

# The sequence changes from one run to another.

# Why is this useful?

# Suppose you're comparing two models.

# Without a fixed seed:

# Model A might see one sequence of shuffled batches.
# Model B might see a different sequence.

# If Model A performs better, you can't be sure whether it's because of the model itself or just because it happened to get a different order of training data.

# With a fixed seed:

# Both models see the exact same batch order across all epochs.
# Any difference in performance is much more likely to be due to the model, not the randomness in data ordering.

# So the seed gives you reproducible experiments, while shuffle=True still ensures that within each run, every epoch presents the data in a fresh random order.

if __name__ == "__main__":
    # print(y_train.shape[0])
    print(train_ds.features)
    print(type(train_loader))

    for idx, (x, y) in enumerate(train_loader): print(f"Batch {idx+1}:", x, y)



