
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 1)
        self.fc2 = torch.nn.Linear(1, 64)
        self.fc3 = torch.nn.Linear(64, 128)
        self.fc4 = torch.nn.Linear(128, 4)
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = F.relu(v1)
        v3 = v2 - 0.5
        v4 = self.fc2(v3)
        v5 = F.relu(v4)
        v6 = v5 - -3.3
        v7 = self.fc3(v6)
        v8 = F.relu(v7)
        v9 = self.fc4(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3)
