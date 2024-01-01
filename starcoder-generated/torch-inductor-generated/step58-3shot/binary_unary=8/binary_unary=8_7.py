
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv1(x1)
        v5 = v1 + v2
        v6 = v3 + v4
        v7 = torch.relu(v5)
        v8 = torch.relu(v6)
        return v7, v8
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
