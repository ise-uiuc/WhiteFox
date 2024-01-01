
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.negative_slope = 0.1
 
    def forward(self, x1):
        v0 = self.linear(x1)
        v1 = v0 > 0
        v2 = v0 * self.negative_slope
        v3 = torch.where(v1, v0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
