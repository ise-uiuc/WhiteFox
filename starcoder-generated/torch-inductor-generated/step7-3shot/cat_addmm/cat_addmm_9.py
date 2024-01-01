
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm = torch.nn.Linear(1024, 3072)
 
    def forward(self, x1, x2, x3):
        v1 = self.gemm(x1)
        v2 = torch.cat([v1], 0)
        if v3 is None:
            return v2
        v4 = torch.cat([v1], 1)
        if v5 is None:
            return v4
        v6 = torch.cat([v2, v4], 0)
        return torch.add(v3, v6)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1024)
x2 = torch.randn(1024)
x3 = None
