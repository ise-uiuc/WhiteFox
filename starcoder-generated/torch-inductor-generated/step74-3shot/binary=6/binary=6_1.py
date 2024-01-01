
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model(128)  # 'other' is a tensor with 128 values in the range [0, 1]

# Inputs to the model
x1 = torch.randn(1, 128)
