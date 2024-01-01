
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v3 = v1[:, 0:9223372036854775807]
        v4 = v3[:, 0:self.size]
        v2 = torch.cat([v1, v4], dim=1)
        return v2

# Initializing the model
m = Model(1)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
