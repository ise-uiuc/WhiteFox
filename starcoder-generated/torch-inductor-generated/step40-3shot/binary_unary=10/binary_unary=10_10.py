
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1.view(1, -1), x2.view(1, -1)], dim = 1) # Concatenate the input tensor on the channel dimension into a tensor with the shape as [1, 2*CHANNELS*H*W]
        v2 = v1.view(-1, 2, 3) # Reshape the tensor to [-1, 2, 3]. -1 means other dimensions that will computed automatically
        v3 = torch.mm(v1.view(2, -1), v1.view(2, -1).T) # Matrix multiplication
        v4 = v3.trace() # Compute the trace of the tensor
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
