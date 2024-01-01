
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Linear(3, 2)
 
    def forward(self, x1):
        v1 = self.f(x1)
        v2 = self.f(x1)
        v3 = torch.sigmoid(v1)
        v4 = torch.sigmoid(v2)
        v5 = v1 * v3
        v6 = v4 * v2
        v7 = torch.add(v5, v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
