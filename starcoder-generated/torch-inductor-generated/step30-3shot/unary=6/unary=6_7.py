
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        d3 = 3
        v1 = self.conv(x1)
        v2 = v1 + d3
        v3 = torch.relu(v2)
        v4 = torch.relu6(v3)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
