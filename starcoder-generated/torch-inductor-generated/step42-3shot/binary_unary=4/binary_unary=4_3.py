
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        _other = torch.randn(1) * 10 # A random tensor with shape of (1,)
        v2 = v1 + _other
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
