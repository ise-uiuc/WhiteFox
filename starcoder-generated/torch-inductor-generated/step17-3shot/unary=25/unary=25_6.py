
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(50, 10)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = v3 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 50)
