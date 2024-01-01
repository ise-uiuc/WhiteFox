
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, split_sizes):
        v1, v2, v3, v4, v5, v6 = torch.split(x1, split_sizes, dim=1) 
        v7 = v1 * 0.5
        v8 = v2 * 0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        v12 = v4 * v10
        v13 = v5 * v10
        v14 = v6 * v10
        v15 = torch.cat([v7, v11, v12, v13, v14], dim=1)
        return v15

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 15, 8, 8, 8)
split_sizes = [2, 3, 1, 7, 2]
