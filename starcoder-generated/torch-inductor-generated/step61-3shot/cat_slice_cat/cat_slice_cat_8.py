
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat([x1, x1, x1], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:171127603233]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 23, 17179869184)
x2 = torch.randn(1, 23, 17179869184)
x3 = torch.randn(1, 23, 17179869184)
x4 = torch.randn(1, 23, 17179869184)
x5 = torch.randn(1, 23, 17179869183)
x6 = torch.randn(1, 23, 17179869183)
x7 = torch.randn(1, 23, 17179869183)
x8 = torch.randn(1, 23, 17179869183)
x9 = torch.randn(1, 23, 17179869182)
