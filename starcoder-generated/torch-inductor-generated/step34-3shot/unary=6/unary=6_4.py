
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 10)
    def forward(self, x1):
        x1 = torch.tanh(self.fc1(x1))
        x2 = self.fc2(x1)
        return x2
# Inputs to the model
x1 = torch.randn(5, 10)
