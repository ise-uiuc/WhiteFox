
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(v1)
        v3 = self.conv1(v2)
        v4 = v1 + v2
        v5 = v1 + v3
        v6 = torch.relu(v4 + v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 8, 8)
