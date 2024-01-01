
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, z):
        v1 = torch.cat([x, 1])
        v2 = torch.cat([v1, 2])
        v3 = v1[0:, 1]
        v4 = v2[1:2, 0:]
        return cat(v4, 1) + torch.slice(v1, 2, 0, 9223372036854775807) * 1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2, 4, 4)
y = torch.randn(2, 1, 4, 4)
z = torch.randn(1, 2)
