
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        x123 = torch.cat((x1, x2, x3), dim=1)
        v1 = x123[:, 0:9223372036854775807]
        v2 = v1[:, 0:6]
        v3 = torch.cat([x123, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
