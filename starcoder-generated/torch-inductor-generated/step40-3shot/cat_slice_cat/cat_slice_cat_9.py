
class Model(torch.nn.Module):
    def __init__(self, size : int):
        super().__init__()
        self.size = size
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:self.size]
        v3 = v2[:,0:self.size]
        x3 = torch.cat([v1,v3], dim=1)
        return x3

# Initializing the model
m = Model(8)

# Inputs to the model
x1 = torch.randn(1, 80, 64, 64)
x2 = torch.randn(1, 90, 64, 64)
