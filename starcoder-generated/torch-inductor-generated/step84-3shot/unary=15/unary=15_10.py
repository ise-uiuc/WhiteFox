
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 2)
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = torch.relu(v1)
        v3 = self.fc2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1024)
