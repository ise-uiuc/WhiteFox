
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._linear1 = torch.nn.Linear(4, 8)
        self._linear2 = torch.nn.Linear(8, 16)
 
    def forward(self, x1):
        v1 = self._linear1(x1)
        v2 = self._linear2(v1)
        v3 = v2.t()
        v4 = v1 + v3
        v5 = torch.cat([v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
