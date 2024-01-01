
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.split([1, 3, 4, 2], 1)
        v2 = v1[v1[1]!= v1[3]].index(4)
        v3 = v1[v2]
        return torch.cat(v3, 1) == x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
