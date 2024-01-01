
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 1)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = self.linear.weight * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m1 = Model(negative_slope=0.5)
m2 = Model(negative_slope=0.2)

# Inputs to the model
x1 = torch.randn(1, 28 * 28)
