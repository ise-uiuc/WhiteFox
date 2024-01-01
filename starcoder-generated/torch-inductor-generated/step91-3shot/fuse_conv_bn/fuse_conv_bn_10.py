
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(2, 2)
        self.bn1 = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(2, 2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.relu(self.bn1(self.lin1(x)))
        x = self.tanh(self.lin2(x))
        return x
# Inputs to the model
x = torch.randn(2, 2)
