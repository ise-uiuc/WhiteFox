
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.5):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.negative_slope = negative_slope
 
    def forward(self, x3):
        v1 = self.linear(x3)
        return torch.where(v1 > 0, v1, v1 * self.negative_slope)

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(4, 16)
