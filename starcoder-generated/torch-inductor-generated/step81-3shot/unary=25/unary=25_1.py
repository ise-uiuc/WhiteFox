
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 300, False)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        t2 = v1 > 0
        t3 = v1 * self.negative_slope
        v4 = torch.where(t2, v1, t3)
        return v4

# Initializing the model
m = Model(negative_slope)

# Inputs to the model
x1 = torch.randn(100, 1000)
