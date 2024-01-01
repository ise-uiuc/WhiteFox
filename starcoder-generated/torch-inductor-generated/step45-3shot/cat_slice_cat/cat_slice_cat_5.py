
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        t0 = torch.cat([x1, x2])
        t1 = t0[:, 0:9223372036854775807]
        t2 = t1[:, 0:9223372036854775807]
        t3 = torch.cat([t0, t2])
        t4 = torch.reshape(t3, [])
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9223372036854775807, 549755813887, 6)
x2 = torch.randn(1, 9223372036854775807, 549755813887, 6)
x3 = torch.randn(1, 9223372036854775807, 549755813887, 6)
x4 = torch.randn(1, 9223372036854775807, 549755813887, 6)
x5 = torch.randn(1, 9223372036854775807, 549755813887, 6)
