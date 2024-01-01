
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 8)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()
m.train()

# Inputs to the model
x1 = torch.randn(5, 1024)
other = torch.rand()
x2 = x1 + other
