
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, None, 0:9223372036854775807]
        v3 = v2[:, 0, 0:self.size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
size = 10
m = Model(size)

# Inputs to the model
x1 = torch.randn(1, 512, 8)
x2 = torch.randn(1, 256, 8)
