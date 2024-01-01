
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.linear(3, 8, bias=False)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        is_positive = v1 > 0
        negative_part = -self.negative_slope * v1
        v2 = torch.where(is_positive, v1, negative_part)
        return v2

# Initializing the model with negative_slope as -1.0
m = Model(-1.0)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
