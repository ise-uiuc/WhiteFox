
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v2 = v2[:, 0:9223372036854775807]
        v3 = torch.cat((v1, v2), dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 4, 16, 64)
