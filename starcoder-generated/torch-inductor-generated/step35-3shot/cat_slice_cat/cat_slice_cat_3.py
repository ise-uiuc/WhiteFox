
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        l1 = torch.cat([x1, x2], dim=1)
        v2 = l1[:, 0:9223372036854775807]
        v0 = v2[:, 0:9223372036854775807]
        l1[:, 0:9223372036854775807] = v0,
        l2 = l1
        return l2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9223372036854775807, 2)
x2 = torch.randn(1, 9223372036854775807, 2)
