
class Model(torch.nn.Module):
    def __init__(self, num_features, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(24)

# Inputs to the model
x1 = torch.randn(1, 24)
