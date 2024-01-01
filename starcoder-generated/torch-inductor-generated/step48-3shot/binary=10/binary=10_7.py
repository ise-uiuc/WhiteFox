
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, other):
        v1 = x1.flatten(1)
        v2 = v1 + other
        v3 = v2.reshape_as(x1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 3, 4, 4)
other = torch.randn(10, 12)
