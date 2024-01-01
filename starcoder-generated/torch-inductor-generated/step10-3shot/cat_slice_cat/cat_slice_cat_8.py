
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1
        v2 = v1
        v3 = v2[:, :, 0:64]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = torch.cat([v4, v4], dim=0)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 3, 128, 128)
x2 = torch.randn(1, 10, 256, 256)
x3 = torch.randn(11, 5, 64, 64)
x4 = torch.randn(3, 1, 512, 512)
x5 = torch.randn(20, 2, 64, 64)
x6 = torch.randn(7, 5, 256, 256)
