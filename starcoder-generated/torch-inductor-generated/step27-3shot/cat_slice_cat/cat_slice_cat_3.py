
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x_in1, x_in2, x_in3):
        c1 = torch.cat([x_in1, x_in2, x_in3], dim=1)
        c2 = c1[:, 0:9223372036854775807]
        s1 = c2[:, 0:7]
        c3 = torch.cat([c1, s1], dim=1)
        return c3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
