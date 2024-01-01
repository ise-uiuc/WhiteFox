
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        h1 = torch.cat([x1, x2], dim=1)
        h2 = h1[:, 0:-9223372036854775802]
        h3 = torch.cat([h1, h2], dim=1)
        return h3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
