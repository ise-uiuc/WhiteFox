
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:x4]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = v4[:, x5]
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 5, 5)
x2 = torch.randn(2, 4, 5, 5)
x3 = torch.randn(2, 4, 5, 5)
x4 = torch.tensor(283746873)
x5 = torch.tensor(283746873)
