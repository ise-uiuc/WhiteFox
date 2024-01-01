
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        linear_features = 10
        negative_slope = 0.1
        self.linear = torch.nn.Linear(linear_features, linear_features)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v0 = self.linear(x1)
        v1 = v0 > 0
        v2 = v0 * self.negative_slope
        v3 = torch.where(v1, v0, v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
