
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 720, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        v1 = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:1120]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = torch.mm(v4, self.weight)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 30320)
x2 = torch.randn(1, 30320)
x3 = torch.randn(1, 30320)
x4 = torch.randn(1, 30320)
x5 = torch.randn(1, 30320)
x6 = torch.randn(1, 30320)
x7 = torch.randn(1, 30320)
x8 = torch.randn(1, 30320)
