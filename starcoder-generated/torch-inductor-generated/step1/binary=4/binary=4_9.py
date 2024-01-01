
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 64)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        return v1 + other

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
other = torch.randn(1, 64)
