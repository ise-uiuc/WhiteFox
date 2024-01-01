
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
        x1  = torch.cat([x2, x3], dim=1)
        x2  = x1[:, 0:4611686018427387903]
        x3  = torch.cat([x4, x5], dim=1)
        x4  = x3[:, 0:4611686018427387903]
        x5  = torch.cat([x6, x7], dim=1)
        x6  = x5[:, 0:4611686018427387903]
        x7  = torch.cat([x8, x9], dim=1)
        x8  = x7[:, 0:4611686018427387903]
        x9  = torch.cat([x10,x11],dim=1)
        x10 = x9[:, 0:4611686018427387903]
        x11 = torch.cat([x1, x2], dim=1)
        return x11

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4611686018427387903, 16)
x2 = torch.randn(4611686018427387903, 16)
x3 = torch.randn(4611686018427387903, 16)
x4 = torch.randn(4611686018427387903, 16)
x5 = torch.randn(4611686018427387903, 16)
x6 = torch.randn(4611686018427387903, 16)
x7 = torch.randn(4611686018427387903, 16)
x8 = torch.randn(4611686018427387903, 16)
x9 = torch.randn(4611686018427387903, 16)
x10 = torch.randn(4611686018427387903, 16)
