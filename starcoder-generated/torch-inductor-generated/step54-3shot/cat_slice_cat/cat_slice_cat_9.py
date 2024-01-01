
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()

    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, :2147483647]
        v3 = v2[:, :self.size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
size = 4294967294
m = Model(size)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 8388608, 64, 64)
x3 = torch.randn(1, 131072, 64, 64)
