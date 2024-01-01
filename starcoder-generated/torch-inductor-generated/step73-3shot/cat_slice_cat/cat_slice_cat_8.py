
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v1, v2, v3):
        x1 = torch.cat([v1, v2, v3], dim=1)
        x2 = x1[:, 0:9223372036854775807]
        x3 = x2[:, 0:3]
        x4 = torch.cat([x1, x3], dim=1)
        return x4, x1

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(shape)
v2 = torch.randn(shape)
v3 = torch.randn(shape)
