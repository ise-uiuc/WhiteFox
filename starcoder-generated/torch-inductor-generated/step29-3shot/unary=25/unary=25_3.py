
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.linear = torch.nn.Linear(-1, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Inputs to the model
negative_slope = -0.1
x1 = torch.randn(2, -1)
m = Model(negative_slope)
