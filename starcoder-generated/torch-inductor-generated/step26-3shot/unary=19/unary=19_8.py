
class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(16, 6)
 
    def forward(self, x1):
      v1 = self.linear(x1)
      vs = torch.sigmoid(v1)
      return vs

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(4, 16)
