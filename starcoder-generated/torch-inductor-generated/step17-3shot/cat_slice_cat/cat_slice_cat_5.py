
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        x2 = torch.cat([x1, x1, x1, x1, x1, x1], dim=1)
        x3 = x2[:, 0:9223372036854775807]
        x4 = x3[:, 0:3]
        x5 = torch.cat([x2, x4], dim=1)
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
