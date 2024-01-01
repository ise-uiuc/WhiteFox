
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
           
    def forward(self, x1, x2, x3, x4):
        t1 = torch.cat([x1, x2, x3, x4], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:65535]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9223372036854775807, 1, 1024)
x2 = torch.randn(1, 8192, 1024, 32)
x3 = torch.randn(1, 65535, 32, 32)
x4 = torch.randn(1, 8192, 32, 64)
