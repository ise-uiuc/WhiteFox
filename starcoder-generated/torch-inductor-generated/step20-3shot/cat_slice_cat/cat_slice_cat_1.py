
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y, z):
        v = torch.cat([x, y, z], dim=1)
        v1 = v[:, 0:9223372036854775807]
        v2 = v1[:, 0:v2.shape[1]]
        v3 = torch.cat([v, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 5)
y = torch.randn(1, 3, 11)
z = torch.randn(1, 3, 25)
