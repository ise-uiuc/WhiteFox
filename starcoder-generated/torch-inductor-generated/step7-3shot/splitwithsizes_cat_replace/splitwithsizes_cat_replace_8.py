
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        x1, x2 = torch.split(x, 2, 3)
        v1 = x1 * 0.5
        v2 = x2 * 0.5
        v3 = torch.cat([v1, v2], 3)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
