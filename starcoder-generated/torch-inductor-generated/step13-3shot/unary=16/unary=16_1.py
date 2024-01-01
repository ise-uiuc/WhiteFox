
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
        self.bn = torch.nn.BatchNorm2d(5)

    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = self.bn(x2)
        x4 = torch.relu(x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 3)
