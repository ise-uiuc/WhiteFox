
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
 
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        t3 = t2[:, 0:size]
        v6 = torch.cat([v1, t3], dim=1)
        return v6
 

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 224, 224)
x2 = torch.randn(1, 16, 112, 112)
x3 = torch.randn(1, 32, 56, 56)
x4 = torch.randn(1, 32, 56, 56)
x5 = torch.randn(1, 64, 28, 28)
x6 = torch.randn(1, 64, 28, 28)
