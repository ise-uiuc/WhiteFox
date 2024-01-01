
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_max(v2, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
