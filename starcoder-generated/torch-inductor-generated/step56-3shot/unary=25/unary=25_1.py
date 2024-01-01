
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
        self.negative_slope = float(negative_slope)
 
    def forward(self, x1):
        x1 = self.linear(x1)
        x2 = x1 > 0
        x3 = x1 * self.negative_slope
        x4 = torch.where(x2, x1, x3)
        return x4

# Initializing the model
m = Model(negative_slope=0.01)

# Inputs to the model
x1 = torch.randn(1, 128)
