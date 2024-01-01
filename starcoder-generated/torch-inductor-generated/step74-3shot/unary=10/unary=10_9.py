
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1)
 
    def forward(self, x1, x2):
        l1 = self.linear(x1)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        v1 = self.conv(l5)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 5, 16, 16)
