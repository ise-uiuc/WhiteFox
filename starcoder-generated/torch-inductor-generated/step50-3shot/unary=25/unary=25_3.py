
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(8, 128, bias=False)
        self.negative_slope = 1e-2
 
    def forward(self, x2):
        v2 = self.layer(x2)
        v3 = v2 > 0
        v4 = v2 * self.negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
