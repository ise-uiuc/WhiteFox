
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(18, 64)
        self.fc2 = torch.nn.Linear(64, 16)
        self.fc3 = torch.nn.Linear(16, 7)
    def forward(self, x1):
        v1 = torch.flatten(x1, start_dim=1)
        v2 = self.fc1(v1)
        v3 = torch.relu(v2)
        v4 = self.fc2(v3)
        v5 = torch.relu(v4)
        v6 = self.fc3(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 18)
