
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:2 * x1.size()[2]]
        v4 = torch.cat([v1, v3], dim=1)
        return (v1 + v2) * v3.mean()

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 128, 64, 64)
x2 = torch.randn(2, 128, 64, 64)
