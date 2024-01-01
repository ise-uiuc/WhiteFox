
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = v2  # dummy variable used to make sure the result of v2 is used in the where() operator
        v4 = v1 * self.negative_slope
        v5 = v3 * v2 + ~v3 * v4
        return v5

# Initializing the model
m = Model(0.1)

# Inputs to the model
x = torch.randn(4, 3)
