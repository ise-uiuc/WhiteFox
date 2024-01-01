
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, other):
        v1 = torch.nn.functional.linear(x1, torch.ones(16, 3, 1, 1))
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 3, 64, 64)
other = torch.randn(16, 16, 64, 64)
