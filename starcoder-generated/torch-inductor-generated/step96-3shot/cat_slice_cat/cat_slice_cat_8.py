
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        t1 = torch.cat([x1, x2, x3], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:2]
        t4 = torch.cat([t1, t3], dim=1)
        t5 = t4[0]
        return t5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
x2 = torch.randn(1, 2, 32, 32)
x3 = torch.randn(1, 2, 16, 16)
x4 = torch.randn(1, 2, 8, 8)
