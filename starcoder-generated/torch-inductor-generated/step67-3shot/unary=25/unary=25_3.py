
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = (v1 >= 0).float()
        v3 = v1 * self.negative_slope
        v4 = v1 * v2 + v3
        return v4

# Initializing the model with a negative slope
m = Model(-0.25)

# Inputs to the model
x1 = torch.randn(1, 2)
