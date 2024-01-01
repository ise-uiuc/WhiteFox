
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = x1.mean(-1)
        v2 = torch.greater(v1, 0)
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
negative_slope = 0.01
m = Model(negative_slope)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
