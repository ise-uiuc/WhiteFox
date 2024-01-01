
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = v2.flatten(1)
        v4 = torch.tanh(self.fc1(v3))
        v5 = torch.tanh(self.fc2(v4))
        v6 = torch.tanh(self.fc3(v5))
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
