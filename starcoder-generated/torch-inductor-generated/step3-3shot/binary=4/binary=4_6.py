
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1, x2, other):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = v2 + other
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 10)
x2 = torch.randn(5, 10)
other = torch.randn(5, 10)
