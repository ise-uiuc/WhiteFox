
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        l1 = torch.cat([x1, x2, x3], dim=1)
        t1 = l1[:, 0:9223372036854775807]
        t2 = t1[:, 0:15]
        l2 = torch.cat([l1, t2], dim=1)
        return l2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20, 25)
x2 = torch.randn(1, 15, 25)
x3 = torch.randn(1, 10, 25)
