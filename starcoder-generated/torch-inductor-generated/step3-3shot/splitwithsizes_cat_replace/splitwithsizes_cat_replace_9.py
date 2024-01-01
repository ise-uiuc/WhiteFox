
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1, v2, v3, v4, v5, v6, v7, v8 = torch.split(x1, [3, 3, 3, 3, 3, 3, 3, 3], 1)
        v9 = torch.cat([v1, v2, v3, v4, v5, v6, v7, v8], 1)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 1, 1)
