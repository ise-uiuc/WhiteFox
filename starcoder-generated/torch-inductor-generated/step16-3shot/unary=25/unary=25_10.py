
class Model(torch.nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.slope = slope
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.conv.weight)
        v2 = v1 > 0
        v3 = v1 * self.slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(0.1)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
