
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat((x1, x2), 1)
        v2 = v1[:, -9223372036854775808:]
        v3 = v2[:, :16]
        v4 = torch.cat((v1, v3), 1)
        v5 = v4 - x3
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
x2 = torch.randn(1, 1, 8, 8)
x3 = torch.randn(1, 1, 4, 4)
