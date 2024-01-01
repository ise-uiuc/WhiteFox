
class Model(torch.nn.Module):

    __constant_size0__ = 9223372036854775807
    __constant_size1__ = 9223372036854775807
    __constant_size2__ = 9223372036854775807

    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, Model.__constant_size0__ :]
        v3 = v2[:, Model.__constant_size1__ :]
        return torch.cat([v1, v3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9223372036854775807)
x2 = torch.randn(1, 9223372036854775807)
x3 = x1 * x2
