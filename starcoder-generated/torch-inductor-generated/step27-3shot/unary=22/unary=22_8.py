
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.fc = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        x1 = self.bn(x1)
        x1 = x1.view((-1,16))
        x1 = self.fc(x1)
        x1 = torch.tanh(x1)
        return x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 16, 31)
