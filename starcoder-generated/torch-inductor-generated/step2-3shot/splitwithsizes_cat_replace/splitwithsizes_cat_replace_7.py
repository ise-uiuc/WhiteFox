
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        v1, v2, v3 = torch.split(x, (8, 8, 8))
        return torch.cat([v1, v2, v3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 28, 40, 40)
