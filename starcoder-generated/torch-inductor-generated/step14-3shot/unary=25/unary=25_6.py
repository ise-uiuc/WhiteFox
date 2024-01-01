
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * neg_slope
        return torch.where(v2, v1, v3)

# Initializing the model
m = Model()
neg_slope = 0.01

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
