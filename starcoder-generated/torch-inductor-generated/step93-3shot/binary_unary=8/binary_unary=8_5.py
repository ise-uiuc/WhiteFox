
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + self.conv1(x1)
        v3 = v2 + torch.relu(self.conv1(x1))
        v4 = torch.relu(v2)
        v5 = self.conv1(x1)
        v6 = self.conv1(x1)
        v7 = torch.relu(v4)
        v8 = v5 + v6 + v7
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
