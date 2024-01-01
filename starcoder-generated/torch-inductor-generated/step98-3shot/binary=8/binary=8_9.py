
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(100, 60)
        self.fc2 = torch.nn.Linear(100, 60)
    def forward(self, x, y):
        v1 = self.fc1(x)
        v2 = self.fc2(y)
        v3 = v1 * v2
        return v3
# Inputs to the model
x = torch.randn(1, 100)
y = torch.randn(1, 100)
