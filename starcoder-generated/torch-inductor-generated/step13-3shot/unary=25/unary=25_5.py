
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.linear = torch.nn.Linear(1, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = self.negative_slope * v1
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
negative_slope = 0.2
m = Model(negative_slope)

# Inputs to the model
x1 = torch.randn(1, 1)
