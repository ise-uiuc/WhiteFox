
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        l = [x1, x2]

        t1 = torch.cat(l, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:13]
        t4 = torch.cat([t1, t3], dim=1)
        return t4, x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 2, 3)
x2 = torch.randn(1, 32, 5, 4)
