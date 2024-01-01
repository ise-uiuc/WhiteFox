
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = v1 * v2 * v3
        v5 = v1 * v2
        v5 += v4
        v4 = torch.relu(v5)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
