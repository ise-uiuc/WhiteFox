
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = 0.1
m = Model(negative_slope)

# Inputs to the model
x = torch.randn(1, 64)
