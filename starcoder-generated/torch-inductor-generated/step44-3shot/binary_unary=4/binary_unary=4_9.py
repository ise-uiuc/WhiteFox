
class Model(torch.nn.Module):
    def __init__(self, other=None):
        super().__init__()
        self.other = other
        self.linear = torch.nn.Linear(64, 8)
 
    def forward(self, x1):
      v1 = self.linear(x1)
      v2 = v1 + self.other
      v3 = torch.nn.functional.relu(v2)
      return v3

# Initializing the model
other = torch.randn(1, 8)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 64)
