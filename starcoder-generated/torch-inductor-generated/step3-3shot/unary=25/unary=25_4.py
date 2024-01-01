
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(64, 100)
 
    def forward(self, x1):
        y = self.layer(x1)
        z = y > 0
        w = y * self.negative_slope
        x = torch.where(z, y, w)
        return x

# Initializing the model
m = Model()
m.negative_slope = 0.1

# Inputs to the model
x1 = torch.randn(1, 64)
