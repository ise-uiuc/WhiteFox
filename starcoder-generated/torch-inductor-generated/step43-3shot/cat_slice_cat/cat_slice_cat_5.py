
class Model(torch.nn.Module):
    def __init__(self):
            super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:10]
        v3 = v2[:, 0:v2.shape[1] - 1]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = torch.cat([x4, x5, x6], dim=1)
        v6 = v5[:, 0:4]
        v7 = v6[:, 0:v6.shape[1] - 2]
        v8 = torch.cat([v5, v7], dim=1)
        v9 = torch.cat([x7, x8, x9], dim=1)
        v10 = v9[:, 0:19]
        v11 = v10[:, 0:v10.shape[1] - 1]
        v12 = torch.cat([v9, v11], dim=1)
        return v4, v8, v12
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10) 
x2 = torch.randn(1, 12) 
x3 = torch.randn(1, 9) 
x4 = torch.randn(1, 15) 
x5 = torch.randn(1, 7) 
x6 = torch.randn(1, 5) 
x7 = torch.randn(1, 19) 
x8 = torch.randn(1, 6) 
x9 = torch.randn(1, 22) 
