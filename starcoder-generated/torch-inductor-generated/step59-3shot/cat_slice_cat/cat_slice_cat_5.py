
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        y = x[:, 0:9223372036854775807]
        z = y[:, 0:9223372036854775807]
        x = torch.cat([x, z], dim=1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 24, 24)
x2 = torch.randn(1, 702351, 24)
x3 = torch.randn(1, 999327, 24)
