
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = v2 + v2
        v4 = self.conv1(x1)
        v5 = torch.relu(v4)
        v6 = v5 + v5
        v7 = self.conv1(x1)
        v8 = torch.relu(v7)
        v9 = v8 + v8
        v10 = v3 + v6 + v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
