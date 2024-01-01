
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(28, 24, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = torch.cat([v1, v2, v2], 1)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 28, 64, 64)
