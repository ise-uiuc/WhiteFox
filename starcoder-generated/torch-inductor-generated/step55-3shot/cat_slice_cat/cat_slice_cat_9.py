
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x2):
        x1 = x2
        h1 = x2[:, :9223372036854775807]
        h2 = h1[:, :9223372036854775807]
        x3 = torch.cat([x1, h2], dim=1)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(3, 9223372036854775807, 3, 3)
