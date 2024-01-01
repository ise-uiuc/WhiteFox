
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=True)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 > 0
        x4 = x2 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        return x5

# Initializing the model with negative_slope = 0.2
m = Model(0.2)

# Inputs to the model
x1 = torch.randn(1, 1)
