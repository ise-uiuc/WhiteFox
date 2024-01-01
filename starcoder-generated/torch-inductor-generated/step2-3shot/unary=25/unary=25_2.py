
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
        self.negative_slope = -0.3
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        return torch.where(v2, v1, v3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
