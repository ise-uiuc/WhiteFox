
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        return v1 + other

# Initializing the model
other = torch.randn(1, 8, 64, 64)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
