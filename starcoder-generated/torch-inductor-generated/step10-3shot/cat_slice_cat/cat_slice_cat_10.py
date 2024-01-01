
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x0, x1):
        v0 = torch.cat([x0, x1], dim=1)
        v1 = v0[:, 0:9223372036854775807]
        v2 = v1[:, 0:x0.size(2)]
        v3 = torch.cat([v0, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 4, 64, 64)
x1 = torch.randn(1, 4, 64, 64)
