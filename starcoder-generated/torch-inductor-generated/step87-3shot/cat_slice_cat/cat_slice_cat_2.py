
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.size = 32
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, : -1]
        v3 = v1[:, -self.size:]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
self.size = 32
x1 = torch.randn(1, 32, 30, 30)
x2 = torch.randn(1, 32, 20, 20)
x3 = torch.randn(1, 32, 10, 10)
x4 = torch.randn(1, 32, 5, 5)
x5 = torch.randn(1, 32, 2, 2)
