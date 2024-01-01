
class Model(torch.nn.Module):
    def __init__(self, linear):
       super().__init__()
       self.linear = linear

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 1
        return v2

linear = torch.nn.Linear(10, 10)

# Initializing the model
m = Model(linear)

# Inputs to the model
x1 = torch.randn(1, 10)
