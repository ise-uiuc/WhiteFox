
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(16, 100, bias=False)
        self.fc2 = torch.nn.Linear(100, 100, bias=False)
        self.fc3 = torch.nn.Linear(100, 10)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.fc1(v3)
        v5 = v4 + x1
        v6 = torch.relu(v5)
        v7 = self.fc2(v6)
        v8 = v7 + x2
        v9 = torch.relu(v8)
        v10 = self.fc3(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
