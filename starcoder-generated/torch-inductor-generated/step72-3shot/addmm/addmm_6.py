
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3)
        self.fc2 = torch.nn.Linear(3, 3)
    def forward(self, x1, x2):
        x = self.fc1(x1)
        x = self.fc2(x)
        return x
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
