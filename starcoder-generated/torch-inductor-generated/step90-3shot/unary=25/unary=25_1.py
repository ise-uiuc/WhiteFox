:
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
 
    def forward(self, x):
        v1 = m(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
