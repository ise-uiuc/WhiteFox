
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat([x1, x2, x3, x4], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v1[:, 0:9223372036854775807]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 16, 16)
x2 = torch.randn(1, 5, 8, 8)
x3 = torch.randn(1, 5, 4, 4)
x4 = torch.randn(1, 5, 2, 2)
