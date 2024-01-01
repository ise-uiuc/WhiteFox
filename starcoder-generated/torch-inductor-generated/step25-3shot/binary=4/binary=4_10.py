
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)
        self.linear = torch.nn.Linear(2, 2)
 
    def forward(self, x1, x2):
        t = self.linear(x1)
        return t + x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.zeros(1, 2)
