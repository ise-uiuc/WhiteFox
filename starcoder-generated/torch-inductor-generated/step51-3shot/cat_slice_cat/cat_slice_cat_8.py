
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = [x1, x2, x3, x4, x5, x6]
        v2 = torch.cat(v1, dim=1)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:x3.size(2)/2]
        return torch.cat([v2, v4], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
x2 = torch.randn(1, 64, 64, 64)
x3 = torch.randn(1, 32, 32, 64)
x4 = torch.randn(1, 32, 32, 64)
x5 = torch.randn(1, 32, 16, 64)
x6 = torch.randn(1, 32, 16, 64)
