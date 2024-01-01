
class Model(torch.nn.Module):
    def forward(self, x1):
        t1 = torch.cat([x1, x1, x1], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:64]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
t1 = torch.randn(1, 3, 64, 64)
t2 = torch.randn(1, 3, 64, 63)
t3 = torch.randn(1, 3, 64, 62)
