
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 32, 3, padding=1)
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.conv(torch.cat([x6,x4,x3,x2,x1], 1))
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 224, 224)
x2 = torch.randn(1, 64, 224, 224)
x3 = torch.randn(1, 64, 224, 224)
x4 = torch.randn(1, 64, 224, 224)
x5 = torch.randn(1, 64, 224, 224)
x6 = torch.randn(1, 64, 224, 224)
