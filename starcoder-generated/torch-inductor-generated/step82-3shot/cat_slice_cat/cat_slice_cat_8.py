
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        e1 = torch.cat([x1, x2, x3], dim=1)
        e2 = e1[:, 0:9223372036854775807]
        e3 = e2[:, 0:10]
        e4 = torch.cat([e1, e3], dim=1)
        return e4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 35, 3)
x2 = torch.randn(1, 5, 3, 7)
x3 = torch.randn(1, 2, 7, 11)
x4 = torch.randn(1, 6, 9, 15)
