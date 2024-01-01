s
class Model1(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the models
m1 = Model1(0.2)
m2 = Model1(0.01)

# Inputs to the models
x1 = torch.randn(7, 2)
x2 = torch.randint(low=-4, high=1, size=(3, 3, 3), dtype=torch.long)
