
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v4 = torch.stack([x1, x2, x3], dim=0)
        v5 = torch.tensor([1.0, 2.0, 3.0]).to(v4.dtype).to(v4.device)
        v6 = v5.size(0)
        v7 = torch.arange(0, v6, dtype=torch.int64).to(v4.device)
        v1 = v4[v7, :v6]
        v2 = v4[:, 0:v6 + 1]
        v10 = torch.tensor([3, 2, 1], dtype=torch.int64).to(v4.device)
        v3 = v4[:, v10]
        v8 = torch.cat([v1, v2, v3], dim=1)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3,5,6)
x2 = torch.randn(3,6,7)
x3 = torch.randn(3,6,8)
