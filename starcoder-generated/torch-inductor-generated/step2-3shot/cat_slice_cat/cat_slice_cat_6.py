
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = torch.cat([x3, x4, x5], dim=1)
        v3 = torch.cat([x6, v1, v2], dim=1)
        v4 = v3[:, 0:9223372036854775807]
        v5 = v4[:, 0:x3.size(1) + x4.size(1)]
        v6 = torch.cat([v1, v5], dim=1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 2, 512)
x2 = torch.randn(2, 3, 512)
x3 = torch.randn(2, 5, 512)
x4 = torch.randn(2, 7, 512)
x5 = torch.randn(2, 11, 512)
x6 = torch.randn(2, 13, 512)
x7 = torch.randn(2, 17, 512)
