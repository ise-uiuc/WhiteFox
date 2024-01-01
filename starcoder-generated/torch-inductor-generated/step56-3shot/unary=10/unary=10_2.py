
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
  
    def forward(self, x1):
        v1 = l1 = linear(x1)
        v2 = l2 = l1 + 3
        v3 = l3 = torch.clamp_min(l2, 0)
        v4 = l5 = torch.clamp_max(l3, 6)
        v5 = l4 = l5 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
