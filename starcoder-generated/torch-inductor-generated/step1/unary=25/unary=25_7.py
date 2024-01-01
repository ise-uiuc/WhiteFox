
class Model(torch.nn.Module):
    def __init__(self, negative_slope: float):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.gt(v1, 0)
        v3 = self.linear(v1)
        v4 = v3 * self.negative_slope
        v5 = torch.where(v2, v3, v4)
        return v5

# Initializing the model
m = Model(negative_slope=0.3)

# Inputs to the model
x = torch.randn(1, 1)
