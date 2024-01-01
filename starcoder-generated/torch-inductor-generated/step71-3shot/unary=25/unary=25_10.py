 
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1, x2):
        v1 = x2 @ x1.T
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
n1 = float(input()) # Initializing negative slope
m = Model(n1)

# Inputs to the model
x1 = torch.randn(32, 3)
x2 = torch.randn(5, 7)
