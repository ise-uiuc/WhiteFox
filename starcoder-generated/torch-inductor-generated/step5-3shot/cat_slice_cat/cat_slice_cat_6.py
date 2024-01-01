
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        p1 = torch.cat([x1, x2], dim=1)
        p2 = p1[:, 0:9223372036854775807]
        p3 = p2[:, 0:64]
        p4 = torch.cat([p1, p3], dim=1)
        return p4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
x2 = torch.randn(1, 10, 64)
