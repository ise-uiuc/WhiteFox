
class Model(torch.nn.Module):
    def __init__(self, nelements, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(nelements, 1, bias=False)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
nelements = 640*3072
negative_slope = 0.1
m = Model(nelements, negative_slope)

# Inputs to the model
x1 = torch.randn(1, 640, 3072)
