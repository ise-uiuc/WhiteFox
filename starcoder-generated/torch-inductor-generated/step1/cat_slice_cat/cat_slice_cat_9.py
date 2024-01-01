
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x0, x1, x2):
        v0 = torch.cat((x0, x1), 1)
        v1 = torch.cat((x0, x1), 0)
        v2 = v1[0:9223372036854775807:1]
        return torch.cat((v0, v2), 1)

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn((3, 5), requires_grad=True)
x1 = torch.randn((5, 3), requires_grad=True)
x2 = torch.randn((3, 5), requires_grad=True)
