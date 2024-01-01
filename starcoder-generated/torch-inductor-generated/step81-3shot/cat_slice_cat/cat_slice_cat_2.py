
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        t1 = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8], dim=1)
        v1 = t1[:, 0:9223372036854775807]
        v2 = v1[:, 0:48]
        v3 = torch.cat([t1, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9333249748535177215, 4, 64, 64)
x2 = torch.randn(1, 222222222222271483, 10, 32, 32)
x3 = torch.randn(1, 444447555356286643, 1, 32, 32)
x4 = torch.randn(1, 444447571556286643, 1, 16, 16)
x5 = torch.randn(1, 444447571556286643, 1, 16, 16)
x6 = torch.randn(1, 444447571556286643, 1, 16, 16)
x7 = torch.randn(1, 444447571556286643, 1, 16, 16)
x8 = torch.randn(1, 444447571556286643, 1, 16, 16)
