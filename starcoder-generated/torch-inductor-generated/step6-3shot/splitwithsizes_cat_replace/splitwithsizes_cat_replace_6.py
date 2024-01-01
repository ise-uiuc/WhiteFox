
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v10, v11, v12 = torch.split(x1, 3, dim=2)
        v21, v22, v23 = torch.split(x1, 3, dim=3)
        v3 = torch.cat([v10, v21], dim=2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 4, 12)
