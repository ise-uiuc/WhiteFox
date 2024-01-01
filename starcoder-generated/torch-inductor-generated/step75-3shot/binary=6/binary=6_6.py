
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        return v2

# Initializing the model with other = torch.tensor(...)
# other represents a Python scalar, not a PyTorch tensor
m = Model()
other = __torch_tensor_to_np_array__(1.23, 'float32')

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.tensor(..., dtype=torch.float32) # The type and shape of 'other' should be consistent
