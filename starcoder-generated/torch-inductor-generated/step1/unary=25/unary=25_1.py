
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 3)
 
    def forward(self, x, negative_slope=0.01):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = torch.where(v2, v1, v1 * negative_slope)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
