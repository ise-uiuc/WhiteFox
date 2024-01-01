
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        negative_slope = 0.1
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(10, 2)
