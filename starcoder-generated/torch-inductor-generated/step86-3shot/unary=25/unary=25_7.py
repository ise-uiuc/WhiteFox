
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.linear(x1)
        v3 = v1 > 0
        v4 = v1 * self.negative_slope
        v5 = torch.where(v3, v1, v4)
        return v5

# Initializing the model
m = Model(0.001)

# Inputs to the model
x1 = torch.randn(1, 2)
