
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = v1 + self.conv1(x1)
        v3 = v1 + self.conv1(x1)
        v4 = self.conv1(x1)
        v5 = v2 + v3 + v4
        v6 = self.conv1(x1)
        v7 = v2 + v3 + v4 + v6
        v8 = self.conv1(x1)
        v9 = v2 + v3 + v4 + v6 + v8
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
