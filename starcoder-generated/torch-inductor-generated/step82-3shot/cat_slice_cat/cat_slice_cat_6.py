
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        a1 = [x1, x2]
        a2 = torch.cat(a1, dim=1)
        a3 = a2[:, 0:9223372036854775807]
        a4 = a3[:, 0:x1.size(2)]
        a5 = torch.cat([a2, a4], dim=1)
        return a5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 288, 64, 64)
x2 = torch.randn(1, 56, 64, 64)
