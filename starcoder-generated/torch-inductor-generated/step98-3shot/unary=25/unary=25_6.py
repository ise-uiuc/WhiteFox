
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 28 * 28)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope_values = [1.0, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

# Initializing the model
for negative_slope in negative_slope_values:
    m = Model(negative_slope)
    x = torch.randn(1, 28 * 28)
    