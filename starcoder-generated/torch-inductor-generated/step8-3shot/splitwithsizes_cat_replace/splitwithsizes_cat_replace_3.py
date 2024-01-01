
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1 + 1
        v2, v3 = torch.split(x1, 4, 1)
        v4, v5 = torch.split(x1, 2, 2)
        v6 = v1 + v2 + v5
        v7 = torch.cat([v1, v5, v6, v4], 2)
        v8 = torch.cat([v4, v3, v2], 1)
        v9 = torch.cat([v2, v1], 1)
        return v1 + v7 + v8 + v9
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 4, 5)
