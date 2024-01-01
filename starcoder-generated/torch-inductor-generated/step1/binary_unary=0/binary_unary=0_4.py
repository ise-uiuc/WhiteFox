
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v2 = 0.5
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v3 = self.v2
        v4 = v1 + v3
        v5 = torch.relu(v4)
        return v5

# Initializing the model
m = Model()
m.v2 = 3

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
