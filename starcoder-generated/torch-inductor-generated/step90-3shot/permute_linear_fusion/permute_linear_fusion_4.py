
self = torch.nn.Linear(3, 3, False)
t1 = input_batch_2d.permute(...) # Permute the input tensor
t2 = self.weight.permute(...) # Permute the weight tensor
t3 = func(t1, t2) # Apply linear transformation to the permuted tensor and return
# Inputs to the model
input_batch_2d = torch.randn(1, 3)
