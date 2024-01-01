
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + 3
        v4 = v3.clamp(0, 6)
        v5 = v1 * v4
        v6 = torch.div(v5, 6)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16, 32, 32)
