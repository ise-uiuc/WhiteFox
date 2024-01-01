
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        v2 = v1
        v3 = v1[:, 0:1073741824]
        v4 = v3[:, 0:107]
        v5 = torch.cat([v2, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 107, 10, 10)
x2 = torch.randn(1, 107, 10, 10)
x3 = torch.randn(1, 107, 10, 10)
x4 = torch.randn(1, 107, 10, 10)
x5 = torch.randn(1, 107, 10, 10)
