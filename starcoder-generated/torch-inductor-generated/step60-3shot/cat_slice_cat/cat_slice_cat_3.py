
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:4194303]
        v3 = v2[:, 0:4194303]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 41006, 81006)
x2 = torch.randn(1, 19000, 27000)
x3 = torch.randn(1, 13000, 31000)
x4 = torch.randn(1, 56535, 23455)
