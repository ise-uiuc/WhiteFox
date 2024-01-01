
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(10, 5)
        self.fc3 = torch.nn.Linear(5, 10)
    def forward(self, x1, x2):
        v1 = self.fc1(x1)
        v2 = self.fc2(x2)
        v3 = v1 + v2
        v4 = self.fc3(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
