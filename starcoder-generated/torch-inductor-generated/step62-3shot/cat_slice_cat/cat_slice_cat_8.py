
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = x[:, 0:9223372036854775807]
        x = x[:, 0:64]
        x = torch.cat([x, x1], dim=1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 125, 56, 56)
x2 = torch.randn(1, 125, 64, 64)
