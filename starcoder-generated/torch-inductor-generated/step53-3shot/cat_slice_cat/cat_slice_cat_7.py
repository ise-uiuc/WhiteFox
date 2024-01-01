
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        y1 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        y2 = y1[:, 0:9223372036854775807]
        y3 = y2[:, 0:10]
        z = torch.cat([y1, y3], dim=1)
        return z, y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
x3 = torch.randn(1, 10)
x4 = torch.randn(1, 10)
x5 = torch.randn(1, 10)
