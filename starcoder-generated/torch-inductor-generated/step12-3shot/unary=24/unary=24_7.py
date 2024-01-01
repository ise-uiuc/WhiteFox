
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
