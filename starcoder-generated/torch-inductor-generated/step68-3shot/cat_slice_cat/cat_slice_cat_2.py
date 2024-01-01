
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        y1 = [y for idx in range(499)]
        y2 = torch.cat(y1[:498], dim=1)
        z = torch.cat([y, y2], dim=1)
        return z

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 24973524783, dtype=torch.float)
