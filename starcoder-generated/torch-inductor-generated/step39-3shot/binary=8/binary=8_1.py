
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Linear(3 * 8 * 32 * 32, 7, bias = False)
        self.bn = torch.nn.BatchNorm1d(7)
    def forward(self, x1, x2, x3):
        v0 = x1.reshape((-1, 3 * 8 * 32 * 32)) # Linear input shape = (3*8*32*32, )
        v1 = self.conv1(v0) # Linear output shape = (7, )
        v2 = x2.reshape((-1, 3 * 8 * 32 * 32)) # Linear input shape = (3*8*32*32, )
        v3 = self.conv1(v2) # Linear output shape = (7, )
        v4 = torch.stack([v1, v3], dim = 0)
        v5 = torch.stack([v3, v1], dim = 0)
        v6 = v4.reshape((-1, 2, 7))
        v7 = self.bn(v6)
        v8 = torch.cat([v7, v5], dim  = 1)
        v9 = v8.reshape((-1, 14, 7))
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
