
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1):
        v1 = x1 + 1
        v2 = torch.split(v1, sizes=(4, 5), dim=0)
        v3 = torch.cat([v2[1], v2[0], v2[1], v2[1]], dim=0)
        v4 = v3 + 2
        return True

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(17, 4, 5)
