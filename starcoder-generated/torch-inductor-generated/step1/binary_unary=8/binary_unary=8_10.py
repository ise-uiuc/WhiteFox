
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x, other):
        v1 = self.conv(x)
        v2 = other + v1
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
o = torch.randn(1, 3, 64, 64)
