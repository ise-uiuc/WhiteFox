
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(20, 30)
  
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3.mean()

# Initializing the model
m = Model(torch.randn(20, 30))

# Inputs to the model
x1 = torch.randn(20, 20)
