
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:x1.size(1)]
        v3 = v2[:, 0:v2.size(1)]
        v4 = torch.cat([x1, v3], dim=1)
        return v4, v4, v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000, 4)
x2 = torch.randn(1, 800, 4)
x3 = torch.randn(1, 125, 4)
