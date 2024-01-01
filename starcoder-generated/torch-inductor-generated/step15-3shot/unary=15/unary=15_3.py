
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)
    def forward(self, x):
        v1 = torch.relu(self.fc1(x))
        v2 = torch.relu(self.fc2(v1))
        v3 = self.fc3(v2)
        return v3
# Inputs to the model
x = torch.randn(10)
