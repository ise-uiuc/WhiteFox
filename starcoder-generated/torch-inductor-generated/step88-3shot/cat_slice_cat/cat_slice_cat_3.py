
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        v1 = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:2]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 288, 512)
x2 = torch.randn(1, 24, 288, 512)
x3 = torch.randn(1, 32, 288, 512)
x4 = torch.randn(1, 64, 144, 256)
x5 = torch.randn(1, 96, 144, 256)
x6 = torch.randn(1, 160, 144, 256)
x7 = torch.randn(1, 320, 72, 128)
x8 = torch.randn(1, 1280, 72, 128)
x9 = torch.randn(1, 1280, 72, 128)
x10 = torch.randn(1, 2560, 72, 128)
