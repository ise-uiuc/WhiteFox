
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(49, 1, 3, padding=16)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        y = self.conv(x)
        z = self.relu(y)
        u = z * self.negative_slope
        v = u - 0.001
        w = v.relu()
        return w
negative_slope = -0.252
# Inputs to the model
x = torch.randn(3, 49, 9, 9)
