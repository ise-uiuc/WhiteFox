
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
  
    def forward(self, x1):
        v1 = self.layer1(x1)
        v2 = v1 * torch.clamp(torch.clamp(v1 + 3.0, 0, 6), 0, 6) / 6.0
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 22, 22)
