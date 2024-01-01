
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.other = other
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.other
        return v1 - v2

# Initializing the model
m = Model(torch.tensor([2, 3]))

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
