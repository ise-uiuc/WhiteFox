
class Model(torch.nn.Module):
  ... 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        return l1 * F.hardtanh(l1 + 3, 0, 6) / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 3)
