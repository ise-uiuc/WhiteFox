
class Model(torch.nn.Module):
    def __init__(self, negative_slope=float('inf')):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        result = torch.where(v2, v1, v3)
        return result

# Initializing the model with the default value of negative_slope
m1 = Model()

# Initializing the model with a negative slope of 0.2
m2 = Model(0.2)

# Inputs to the model
x1 = torch.randn(1, 100)
