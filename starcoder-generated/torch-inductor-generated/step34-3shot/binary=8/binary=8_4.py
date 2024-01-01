
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 256, 1, stride=1, padding=1)
 
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)
other = torch.randn(1, 256, 64, 64)
