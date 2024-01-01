
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

 
    def forward(self, x1, x2, x3):
        c1 = torch.cat((x1, x2, x3), dim=1) 
        c2 = c1[:, 0:18446744073709551615]
        c3 = c2[:, 0:68]
        c4 = torch.cat([c1, c3], dim=1)
        return c1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 32, 32)
