
class Model(torch.nn.Module):
  ...
    def __init__(self):
      ...
        self.linear = torch.nn.Linear(3, 16, bias=True)
        self.bias = torch.nn.Parameter(torch.zeros(16))
 
    def forward(self, x1, y1):
        v1 = self.linear(x1)
        v2 = self.bias + y1
        v3 = torch.relu(v1 + v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
y1 = torch.randn(16)
