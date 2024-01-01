
class Model(torch.nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.size = 2
        self.dim = 1
    
    def forward(self, x1, x2, x3, x4):
        t1 = torch.cat([x1, x2, x3, x4], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:self.size]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = x1[0:1]
x3 = x1[1:2]
x4 = x1[2:3]
