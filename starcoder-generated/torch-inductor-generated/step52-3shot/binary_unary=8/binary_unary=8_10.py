
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1000, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(x1)
        v3 = self.fc2(x1)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1000, 1000)
