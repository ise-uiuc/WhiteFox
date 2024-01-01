
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3 * 28 * 28, 320)
        self.fc2 = torch.nn.Linear(320, 10)
    def forward(self, x1):
        v1 = x1.view(-1, 3, 28, 28)
        v2 = self.fc1(v1)
        v3 = self.fc2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(8, 3, 28, 28)
