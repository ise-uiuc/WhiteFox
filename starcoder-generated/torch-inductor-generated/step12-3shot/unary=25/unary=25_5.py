
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super(Model, self).__init__()
        self.negative_slope = negative_slope
        self.linear = torch.nn.Linear(1024, 10000)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model with negative slope 0.25
m = Model(0.25)

# Inputs to the model
x1 = torch.randn(1024)
