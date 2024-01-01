
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        negative_slope = 0.1
        b = v1 > 0
        v2 = torch.where(b, v1, v1 * negative_slope)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
