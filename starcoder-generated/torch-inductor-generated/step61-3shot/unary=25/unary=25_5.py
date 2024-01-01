
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.m = torch.nn.Linear(5, 3)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.m(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(0.2)

# Inputs to the model
x1 = torch.randn(1, 5)
