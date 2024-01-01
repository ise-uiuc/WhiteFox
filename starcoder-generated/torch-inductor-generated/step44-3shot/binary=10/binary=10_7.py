
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 10)
 
    def forward(self, x, other):
        v = self.linear(x)
        return v + other

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 10)
