
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * self.negative_slope
        v3 = self.linear(x1)
        v4 = v1 > 0
        v5 = torch.where(v4, v1, v2)
        return v5

# Initializing the model for a negative slope of 1/1000
m = Model(1/1000)

# Initializing the model for a negative slope of 0.25
m = Model(0.25)

# Initializing the model for a negative slope of 2
m = Model(2)

# Inputs to the model for each case
x1 = torch.randn(1, 1)
