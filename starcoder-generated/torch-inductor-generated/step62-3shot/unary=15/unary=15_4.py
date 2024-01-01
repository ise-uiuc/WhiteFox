
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
    def forward(self, x1):
        x1 = x1.view(x1.shape[0], -1)
        v1 = self.fc(x1)
        v2 = torch.relu(v1)
        v3 = self.fc1(v2)
        v4 = torch.relu(v3)
        v5 = self.fc2(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(16, 1, 28, 28)
