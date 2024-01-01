
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28, 28)
        self.fc2 = torch.nn.Linear(28, 28)
        self.fc3 = torch.nn.Linear(28, 28)
        self.fc4 = torch.nn.Linear(28, 28)
    def forward(self, x1, x2, x3, x4):
        v1 = self.fc1(x1)
        v2 = self.fc2(x1)
        v3 = self.fc3(x1)
        v4 = self.fc4(x1)
        v5 = v1 + x2
        v6 = torch.relu(v5)
        v7 = v2 + v6
        v8 = torch.relu(v7)
        v9 = v3 + v8
        v10 = torch.relu(v9) + x3
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
