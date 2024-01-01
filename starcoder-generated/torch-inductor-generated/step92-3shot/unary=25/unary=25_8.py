
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=True)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(0.1)

# Inputs to the model
x1 = torch.randn(1, 3)
