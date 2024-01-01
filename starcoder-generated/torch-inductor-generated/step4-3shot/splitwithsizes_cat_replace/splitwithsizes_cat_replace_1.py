
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1, v2 = torch.split(x1, [10, 10], dim=1)
        v3, v4 = torch.split(v2, [5, 5], dim=1)
        v5 = torch.cat([v1, v3, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 60)
