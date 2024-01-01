
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        y1 = torch.cat([x1, x2], dim=1)
        y2 = y1[:, 0:2305843009213693951]
        y3 = y2[:, 0:1]
        y4 = torch.cat([y1, y3], dim=1)
        return y1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
x2 = torch.randn(4, 3, 32, 32)
