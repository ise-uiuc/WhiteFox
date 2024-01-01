
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
  
    def forward(self, x1):
        v2 = torch.cat([x1, x1], dim=1)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:63882500]
        v5 = torch.cat([v2, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 10)
x2 = torch.randn(1, 6, 10)
x3 = torch.randn(1, 4, 10)
x4 = torch.randn(1, 6, 10)
x5 = torch.randn(1, 2, 10)
x6 = torch.randn(1, 30, 10)
x7 = torch.randn(1, 7, 10)
x8 = torch.randn(1, 2, 10)
x9 = torch.randn(1, 9, 10)
x10 = torch.randn(1, 2, 10)
