
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.alpha = negative_slope
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.randn(3 * 3 * 8).reshape(3, 3, 8), bias=None)
        v2 = v1 > 0
        v3 = v1 * self.alpha
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
