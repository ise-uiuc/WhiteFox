
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat(list, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:56]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 56, 56)
x2 = torch.randn(1, 1, 224, 224)
x3 = torch.randn(1, 1, 10, 10)
x4 = torch.randn(1, 1, 92, 92)
list = [x1, x2, x3, x4]
