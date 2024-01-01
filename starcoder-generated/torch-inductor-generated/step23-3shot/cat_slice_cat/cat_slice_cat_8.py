
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x3):
        c2 = torch.cat([x3, x3, x3], dim=1)
        o2 = c2[:, 0:9223372036854775807]
        o3 = o2[:, 0:11]
        c3 = torch.cat([c2, o3], dim=1)
        return c3

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 3, 11, 11)
