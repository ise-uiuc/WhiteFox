
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], 1)
        v2 = v1[:, -9223372036854775808:-1]
        v3 = v2[:, 0:255]
        v4 = torch.cat([v1, v3], 1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 54321, 1234)
x2 = torch.randn(1, 3, 54321, 1234)
x3 = torch.randn(1, 3, 54321, 1234)
