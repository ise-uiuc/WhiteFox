
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x3, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:3141]
        v4 = torch.cat([v3, x1, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3141, 100, 100)
x2 = torch.randn(3141, 100, 100)
x3 = torch.randn(44, 100, 100)
