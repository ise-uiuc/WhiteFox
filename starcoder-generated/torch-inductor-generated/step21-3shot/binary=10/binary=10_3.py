
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)

# Other tensors to the model; specified by the keyword argument "other"
x2 = torch.randn(1, 8)
x3 = torch.randn(1, 8)
x4 = torch.randn(1, 8)

x5 = torch.cat([
    x2,
    x3,
    x4,
])

