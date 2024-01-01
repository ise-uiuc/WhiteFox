
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 6)
 
    def forward(self, x3, other):
        v1 = self.linear(x3)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 2)
other = torch.randn(1, 6)
