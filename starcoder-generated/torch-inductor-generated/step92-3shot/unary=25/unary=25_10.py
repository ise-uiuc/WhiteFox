
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.5):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(0.5)

# Inputs to the model
x1 = torch.randn(1, 3)
