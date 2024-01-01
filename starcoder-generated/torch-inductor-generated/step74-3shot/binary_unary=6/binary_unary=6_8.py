
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.5
        v3 = v1 - 1
        v4 = v1 - 3
        v5 = v1 - 0.7853981633974483
        v6 = v1 - 0.07615941559557649
        v7 = v1 - 2
        v8 = torch.max(v2, v3)
        v9 = torch.max(v4, v5)
        v10 = torch.max(v6, v7)
        v11 = torch.max(v8, v9)
        v12 = torch.max(v10, v11)
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
