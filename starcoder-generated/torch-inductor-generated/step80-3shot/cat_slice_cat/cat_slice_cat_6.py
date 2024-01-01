
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat([x1], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:1073741824]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 1024, 32, 512)
x2 = torch.randn(1, 5, 1024, 128, 256)
