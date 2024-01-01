
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randn(8, 3)
        bias = torch.randn(8)
        self.linear = torch.nn.Linear(3, 8, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * negative_slope(v2)
        return torch.where(v2, v1, v3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
