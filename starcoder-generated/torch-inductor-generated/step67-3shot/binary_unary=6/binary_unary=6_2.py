
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = v2 - other
        v4 = v3 - other
        v5 = v4 - other
        v6 = v5 - other
        v7 = v6 - other
        v8 = v7 - other
        v9 = v8 - other
        v10 = v9 - other
        v11 = v10 - other
        v12 = v11 - other
        v13 = v12 - other
        v14 = v13 - other
        v15 = v14 - other
        return v15

# Initializing the model
m = model()

# Inputs to the model
x1 = torch.randn(1, 10)
