
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v2 = torch.cat([x1, x2, x3], dim=1)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:v1.shape[2]]
        v5 = torch.cat([v2, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 16, 16)
x2 = torch.randn(1, 32, 24, 24)
x3 = torch.randn(1, 32, 32, 32)
