
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2])
        v2 = v1[:, 0:y]
        v3 = v2[:, 0:x]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Generating model
input = ((1, 2), (1, 2), (32, 32))
m = Model()
x1, x2, x3 = [torch.rand(i, dtype=torch.float32) for i in input]
__m__ = (m, (x1, x2, x3))

