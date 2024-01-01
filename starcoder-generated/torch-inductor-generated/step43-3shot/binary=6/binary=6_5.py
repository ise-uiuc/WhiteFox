
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(800, 10)
        self.other = torch.nn.Parameter(torch.ones((10,)))
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 800)
