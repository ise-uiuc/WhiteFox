
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convm = torch.nn.Conv2d(10, 10, 1, stride=1, padding=1)
        self.conv  = torch.nn.Conv2d(10, 10, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        z1 = self.convm(x1)
        z2 = self.conv(x2)
        z3 = torch.cat([z1, z2])
        z4 = torch.sigmoid(z3)
        return z4
# Inputs to the model
x1 = torch.randn(1, 10, 5, 5)
x2 = torch.randn(1, 10, 1, 1)
