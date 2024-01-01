
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        h1 = torch.cat([x1, x2], dim=1)
        h2 = h1[:, 0:9223372036854775807]
        h3 = h2[:, 0:size]
        h4 = torch.cat([h1, h3], dim=1)
        return h4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(1, 9223372036854775807, 1, 1)
h1 = torch.cat([x1, x2], dim=1)
h3 = h1[:, 0:size]
h4 = torch.cat([h1, h3], dim=1)
