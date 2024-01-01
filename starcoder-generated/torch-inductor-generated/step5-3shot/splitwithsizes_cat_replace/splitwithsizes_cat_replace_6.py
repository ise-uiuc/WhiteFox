
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1, v2, v3, v4 = torch.split(x1, [1, 1, 8, 8], 3)
        v5 = torch.cat([v2, v4, v1, v3], 3)
        return True

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
