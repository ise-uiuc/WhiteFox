
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1, x2):
        v1 = self.conv(x1 + x2)
        v2 = v1 * 0.5
        v3 = v1 + x2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
