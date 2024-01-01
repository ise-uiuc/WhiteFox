
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 4)
        self.negative_slope = negative_slope
 
    def forward(self, x1, x2, x3):
        v1 = self.linear1(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.linear1(x2)
        v6 = v5 > 0
        v7 = v5 * self.negative_slope
        return torch.where(v6, v5, v7) + v4

# Initializing the model
m = Model(0.2)

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 4)
x3 = torch.randn(1, 4)
