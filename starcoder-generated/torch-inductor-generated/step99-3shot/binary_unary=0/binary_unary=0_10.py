
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = x1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv1(v4)
        v6 = v1 + v3
        v7 = torch.relu(v6)
        v8 = self.conv1(v7)
        return v8 * v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
