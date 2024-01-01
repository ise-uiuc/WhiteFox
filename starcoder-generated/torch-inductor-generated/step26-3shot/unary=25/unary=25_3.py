
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(26, 13)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

negative_slope = random.uniform(-1, -0.5)

# Initializing the model
m = Model(negative_slope)

# Inputs to the model
x1 = torch.ones(1, 26)
