
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear_ = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        v1 = self.linear_(x1)
        v2 = self.linear_(x1)
        v3 = v2 > 0
        v4 = v1 * self.negative_slope
        v5 = torch.where(v3, v1, v4)
        return v5

# Initializing the model
m = Model(negative_slope = 0.05)

# Inputs to the model
x1 = torch.randn(1, 8)
