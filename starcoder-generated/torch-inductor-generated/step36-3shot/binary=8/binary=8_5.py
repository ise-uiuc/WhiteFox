
t1 = torch.sum(torch.randn(1, 3, 21, 21), (2, 3), keepdim=True) # Apply pointwise convolution with kernel size 1 to the input tensor
t2 = t1 + torch.sqrt(x2) # Add another tensor to the output of the convolution
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
