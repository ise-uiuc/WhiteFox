
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat([x1], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:100]
        return v3, v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 1, 1)
__output__, __concat__ = m(x1)

