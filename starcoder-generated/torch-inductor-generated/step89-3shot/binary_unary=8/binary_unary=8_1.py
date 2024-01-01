
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv1(v4)
        v6 = torch.relu(v5)
        v7 = v3 + v5
        v8 = torch.relu(v7)
        v9 = self.conv1(v8)
        v10 = torch.relu(v9)
        v11 = v6 + v10
        v12 = torch.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 15, 15)
