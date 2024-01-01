
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat([x1, x2, x3, x4], 1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:224]
        return torch.cat([v1, v3], 1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 96, 1, 1)
x2 = torch.randn(1, 256, 1, 1)
x3 = torch.randn(1, 256, 1, 1)
x4 = torch.randn(1, 416, 1, 1)
