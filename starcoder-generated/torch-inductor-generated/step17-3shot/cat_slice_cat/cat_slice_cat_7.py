
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3): # Inputs of __init__ must be removed
        t1 = torch.cat([x1, x2, x3], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:17]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
x2 = torch.randn(1, 196608, 1, 1)
x3 = torch.randn(1, 2048, 1, 1)
