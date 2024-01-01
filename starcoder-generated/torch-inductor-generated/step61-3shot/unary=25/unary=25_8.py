
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=True)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = torch.where(v2, self.linear(x1), self.negative_slope * v1)
        return v3

# Initializing the model
m1 = Model(-0.5)
m2 = Model(-2)

# Inputs to the model
x1 = torch.randn(1, 8)
