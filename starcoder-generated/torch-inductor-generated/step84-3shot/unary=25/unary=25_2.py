
negative_slope = 0.02
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, np.random.randn(64, 64), bias=None)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 3, 224, 224)
