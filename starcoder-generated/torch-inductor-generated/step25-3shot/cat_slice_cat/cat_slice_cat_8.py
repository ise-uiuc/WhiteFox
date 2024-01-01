
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        cat = torch.cat([x1, x2, x3], dim=1)
        t = cat[:, 0:9223372036854775807]
        u = cat[:, 0:8]
        v = torch.cat([cat, u], dim=1)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
x2 = torch.randn(1, 2, 3, 3)
x3 = torch.randn(1, 3, 3, 3)
