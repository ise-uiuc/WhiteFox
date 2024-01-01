
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = x1.mean([2, -1]).mean([2, -1])
        v2 = self.negative_slope
        v3 = v2 * v1
        v4 = v3 if v3 > 0 else x1
        return v4

# Initializing the model
m = Model(negative_slope=0.05)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
