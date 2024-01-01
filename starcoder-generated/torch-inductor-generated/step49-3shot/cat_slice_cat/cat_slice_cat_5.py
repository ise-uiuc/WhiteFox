
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat((x1, x2), 1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:x3.size(1)]
        v4 = torch.cat((v1, v3), 1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 55, 34, 48)
x2 = torch.randn(1, 56, 32, 48)
x3 = torch.randn(1, 27, 12, 48)
