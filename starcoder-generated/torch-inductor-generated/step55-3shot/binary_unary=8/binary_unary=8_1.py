
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv1(x1)
        v6 = self.conv1(x1)
        v7 = v5 + v6
        v8 = torch.relu(v7)
        v9 = v4 + v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
