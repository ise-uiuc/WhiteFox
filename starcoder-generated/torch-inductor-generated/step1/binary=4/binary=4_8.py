
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x, other):
        return self.linear(x) + other

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
other = torch.ones(1, 8)
