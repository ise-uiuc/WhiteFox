
class Model(torch.nn.Module):
    def __init__(self, other=0.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.other = other
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - self.other
        y = torch.relu(v2)
        return y

# Initializing the model
m = Model(0.01)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
