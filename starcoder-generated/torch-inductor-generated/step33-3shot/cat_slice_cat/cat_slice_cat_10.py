
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1_1 = torch.cat([x1, x2, x3, x4], dim = 1)
        v1 = v1_1[:, 0:9223372036854775807]
        v2 = v1[:, 0:size]
        return torch.cat([v1_1, v2], dim = 1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9223372036854775807, 2, 2)
x2 = torch.randn(1, 9223372036854775807, 2, 2)
x3 = torch.randn(1, 9223372036854775807, 2, 2)
x4 = torch.randn(1, 9223372036854775807, 2, 2)
x5 = torch.randn(1, 9223372036854775807, 2, 2)
x6 = torch.randn(1, 9223372036854775807, 2, 2)
