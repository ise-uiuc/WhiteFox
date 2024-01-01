
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8 * 8, 32 * 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 1.0
        v3 = v1 - 1.5
        v4 = v1 - 2.0
        v5 = v1 - 2.5
        v6 = v1 - 3.0
        v7 = torch.max(v2, v3)
        v8 = torch.max(v4, v5)
        v9 = torch.max(v6, v7)
        v10 = torch.max(v6, v8)
        v11 = torch.max(v9, v10)
        v12 = torch.max(v9, v11)
        v13 = torch.max(v12, v11)
        v14 = torch.max(v12, v13)
        return v14

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 8, 8)
