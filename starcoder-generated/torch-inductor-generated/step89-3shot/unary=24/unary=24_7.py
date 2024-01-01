
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        negative_slope = -1.8924019
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 7, 10)
