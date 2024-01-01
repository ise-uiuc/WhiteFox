
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 64)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
# Inputs to the model
x = torch.randn(1, 64)
