
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1,x2,x3):
        v1 = torch.cat([x1,x2,x3], dim=1)
        v1 = v1[:, 0:9223372036854775807] # v1 is sliced
        v1 = v1[:, 0:64] # v1 is sliced again
        v2 = torch.cat([x1,v1], dim=1)
        return v1,v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(1, 3, 32, 32)
__output__, 