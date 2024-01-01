
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 32)
        self.layers_1 = nn.Linear(32, 64)
        self.layers_2 = nn.Linear(64, 128)
        self.layers_3 = nn.Linear(128, 256)
        self.stack = torch.stack
    def forward(self, x):
      x = self.stack([x], dim=1)
      x = self.layers(x)
      x = self.stack([x], dim=1)
      x = self.layers_1(x)
      x = self.stack([x], dim=1) 
      x = self.layers_2(x)
      x = self.stack([x], dim=1)
      x = self.layers_3(x)
      x = torch.stack((x, x, x), dim=2)
      x = x.max(2)[0] # maximum across the columns
      return x
# Inputs to the model
x = torch.randn(2, 2)
