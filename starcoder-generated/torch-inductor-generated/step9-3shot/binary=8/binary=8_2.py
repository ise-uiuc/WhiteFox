
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other = other
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model(torch.randn(1))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
