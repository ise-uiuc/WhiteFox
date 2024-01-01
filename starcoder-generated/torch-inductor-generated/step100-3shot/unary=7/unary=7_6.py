
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 10)
        self.bn = torch.nn.BatchNorm1d(10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(self.linear(x1) + 3, min=0, max=6)
        v3 = v1 * v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 20)
